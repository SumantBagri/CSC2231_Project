import math
from typing import List, Optional, Tuple, Union

import torch


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def reverse_1d_tensor(arr: torch.Tensor) -> torch.Tensor:
    return torch.flip(arr, (0,))


class EulerAncestralDiscreteScheduler:
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[List[float]] = None,
        prediction_type: str = "epsilon",
        *args,
        **kwargs,
    ):
        self.num_train_timesteps = num_train_timesteps

        self.betas = (
            torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_train_timesteps,
                dtype=torch.float32,
            )
            ** 2
        )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        sigmas = torch.tensor(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = torch.cat([reverse_1d_tensor(sigmas), torch.tensor([0.0])]).to(
            dtype=torch.float32
        )
        self.sigmas = sigmas

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = self.sigmas.max()

        # setable values
        self.num_inference_steps = None
        self.timesteps = reverse_1d_tensor(
            torch.linspace(
                0, num_train_timesteps - 1, num_train_timesteps, dtype=torch.float32
            )
        )
        self.is_scale_input_called = False

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        self.is_scale_input_called = True
        return sample

    def interp(
        self, x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
    ) -> torch.Tensor:
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])

        indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indicies = torch.clamp(indicies, 0, len(m) - 1)

        return m[indicies] * x + b[indicies]

    def set_timesteps(
        self, num_inference_steps: int, device: Union[str, torch.device] = None
    ):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps

        timesteps = reverse_1d_tensor(
            torch.linspace(
                0,
                self.num_train_timesteps - 1,
                num_inference_steps,
                dtype=torch.float32,
            )
        )
        sigmas = torch.tensor(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = self.interp(timesteps, torch.arange(0, len(sigmas)), sigmas)
        sigmas = torch.cat([sigmas, torch.tensor([0.0])]).to(dtype=torch.float32)
        self.sigmas = sigmas
        self.timesteps = timesteps

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Tuple:
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        pred_original_sample = sample - sigma * model_output

        sigma_from = self.sigmas[step_index]
        sigma_to = self.sigmas[step_index + 1]
        sigma_up = (
            sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
        ) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device
        noise = torch.randn(
            model_output.shape,
            dtype=model_output.dtype,
            device=device,
        )

        prev_sample = prev_sample + noise * sigma_up

        return prev_sample, pred_original_sample

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        schedule_timesteps = self.timesteps
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = self.sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.num_train_timesteps

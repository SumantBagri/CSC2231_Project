from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from .model.clip import CLIPTextModel
from .model.tokenizer import CLIPTokenizer
from .model.autoencoder import AutoencoderKL
from .model.unet_conditional import UNet2DConditionModel
from .model.dpmsolver_scheduler import DPMSolverMultistepScheduler


class StableDiffusionPipeline(nn.Module):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__()

        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.vae_scale_factor = 2 ** (len(self.vae.block_out_channels) - 1)

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        *args,
        **kwargs,
    ):
        batch_size = 1

        text_input = self.tokenizer(prompt)

        attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:

            uncond_tokens = ""
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(uncond_tokens)
            padding = torch.LongTensor([self.tokenizer.end_of_text]).repeat(
                (1, max_length - uncond_input.shape[1])
            )
            uncond_input = torch.cat([uncond_input, padding], dim=-1)

            attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents):
        latents = 1 / self.vae.scaling_factor * latents
        image = self.vae.decode(latents)
        return image

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = torch.randn(shape, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def forward(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.sample_size * self.vae_scale_factor
        width = width or self.unet.sample_size * self.vae_scale_factor

        # Define call parameters
        batch_size = 1

        device = self.unet.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            latents,
        )

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(timesteps):

            print(f"Timestep: {t.item()}\r", end="")

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )

        # Post-processing
        image = self.decode_latents(latents)

        image = image.permute(0, 2, 3, 1)
        image = (image + 1) * 127.5
        image = image.clamp(0, 255)

        return image

import torch

from .model.unet import UNet2DModel
from .model.vq import VQModel
from .model.ddim_scheduler import DDIMScheduler


class LDMPipeline(torch.nn.Module):
    def __init__(self, vqvae, unet, scheduler):
        super().__init__()
        self.unet = unet
        self.vqvae = vqvae
        self.scheduler = scheduler
        self.num_inference_steps = 20

    @torch.no_grad()
    def forward(self, noise):
        # noise = torch.randn((1, 3, 64, 64))

        # set inference steps for DDIM
        self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)

        with torch.no_grad():
            image = noise
            for t in self.scheduler.timesteps:
                print(f"Timestep = {t.item()}\r", end="")

                residual = self.unet(image, t)

                # compute previous image x_t according to DDIM formula
                prev_image = self.scheduler.step(residual, t, image, eta=0.0)[0]

                # x_t-1 -> x_t
                image = prev_image

                # decode image with vae
            image = self.vqvae.decode(image)

            # process image
            image_processed = image.permute(0, 2, 3, 1)
            image_processed = (image_processed + 1.0) * 127.5
            image_processed = image_processed.clamp(0, 255)

            return image_processed

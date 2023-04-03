import torch
import torch.onnx

from ldm_uncond.latent_diffusion_uncond import LDMPipeline

if __name__ == '__main__':
    diffusion_pipeline = LDMPipeline()

    DTYPE = torch.float16
    DEVICE = torch.device('cuda')

    diffusion_pipeline = diffusion_pipeline.to(device=DEVICE, dtype=DTYPE)

    diffusion_pipeline.export_unet_to_onnx()
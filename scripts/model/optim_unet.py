#!/usr/bin/env python3

import sys
import torch
import torch.onnx

from ldm_uncond.latent_diffusion_uncond import LDMPipeline

if __name__ == '__main__':
    diffusion_pipeline = LDMPipeline()

    DTYPE = torch.float16
    DEVICE = torch.device('cuda')

    diffusion_pipeline = diffusion_pipeline.to(device=DEVICE, dtype=DTYPE)

    if len(sys.argv) > 1:
        diffusion_pipeline.export_unet_to_onnx(fname=sys.argv[1])
    else:
        diffusion_pipeline.export_unet_to_onnx()
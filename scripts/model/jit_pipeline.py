#!/usr/bin/env python3

import sys
import torch
import torch.onnx

from ldm_uncond.latent_diffusion_uncond import LDMPipeline

if __name__ == '__main__':
    # Init inputs and model
    if sys.argv[1] == "fp16":
        DTYPE = torch.float16
    elif sys.argv[1] == "fp32":
        DTYPE = torch.float32
    else:
        print("Missing dtype argument")
        exit(1)

    DEVICE = torch.device('cuda')

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)
    diffusion_pipeline = LDMPipeline().to(device=DEVICE, dtype=DTYPE)
    diffusion_pipeline.eval()

    print("Tracing pipeline")
    torchscript_model = torch.jit.trace(diffusion_pipeline, noise)
    print("Saving traced pipeline")
    torch.jit.save(torchscript_model, f"uldm_jit_{sys.argv[1]}.ptl")

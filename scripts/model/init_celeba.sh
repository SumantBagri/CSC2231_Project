#!/usr/bin/env bash

echo Downloading UNet weights for CelebA...
wget -O ./ldm_uncond/ldm-celebahq-256/unet/diffusion_pytorch_model.bin https://huggingface.co/CompVis/ldm-celebahq-256/resolve/main/unet/diffusion_pytorch_model.bin

echo Downloading VQVAE weights for CelebA...
wget -O ./ldm_uncond/ldm-celebahq-256/vqvae/diffusion_pytorch_model.bin https://huggingface.co/CompVis/ldm-celebahq-256/resolve/main/vqvae/diffusion_pytorch_model.bin
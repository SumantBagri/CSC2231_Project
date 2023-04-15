# wget -O ./ldm_uncond/ldm-celebahq-256/unet/diffusion_pytorch_model.bin https://huggingface.co/CompVis/ldm-celebahq-256/resolve/main/unet/diffusion_pytorch_model.bin
# wget -O ./ldm_uncond/ldm-celebahq-256/vqvae/diffusion_pytorch_model.bin https://huggingface.co/CompVis/ldm-celebahq-256/resolve/main/vqvae/diffusion_pytorch_model.bin

import json
from collections import OrderedDict
from ldm_uncond.latent_diffusion_uncond import LDMPipeline
from ldm_uncond.model.unet import UNet2DModel
from ldm_uncond.model.vq import VQModel
from ldm_uncond.model.ddim_scheduler import DDIMScheduler
import torch
import re
import time


## Initialize models and load weights

unet_root = "ldm_uncond/ldm-celebahq-256/unet/"
vqvae_root = "ldm_uncond/ldm-celebahq-256/vqvae/"
scheduler_root = "ldm_uncond/ldm-celebahq-256/scheduler/"

with open(unet_root + "config.json", "r") as f:
    unet_config = json.load(f)
with open(vqvae_root + "config.json", "r") as f:
    vqvae_config = json.load(f)
with open(scheduler_root + "scheduler_config.json", "r") as f:
    scheduler_config = json.load(f)


unet = UNet2DModel(**unet_config)
vqvae = VQModel(**vqvae_config)
scheduler = DDIMScheduler(**scheduler_config)
# scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)


unet_state_dict = torch.load(
    unet_root + "diffusion_pytorch_model.bin", map_location="cpu"
)
unet_state_dict_new = OrderedDict()

for k, v in unet_state_dict.items():
    if k.startswith("mid_block.resnets.0"):
        name = k.replace("resnets.0", "resnets_0")
    elif k.startswith("mid_block.resnets"):
        i = int(k[18])
        name = k[:18] + str(i - 1) + k[19:]
    else:
        name = k
    unet_state_dict_new[name] = v

unet.load_state_dict(unet_state_dict_new)


vqvae_state_dict = torch.load(
    vqvae_root + "diffusion_pytorch_model.bin", map_location="cpu"
)
vqvae_state_dict_new = OrderedDict()

for k, v in vqvae_state_dict.items():
    if "mid_block.resnets.0" in k:
        name = k.replace("resnets.0", "resnets_0")
    elif "mid_block.resnets" in k:
        name = re.sub(
            r"resnets.(\d+)", lambda match: "resnets." + str(int(match.group(1)) - 1), k
        )
    else:
        name = k
    vqvae_state_dict_new[name] = v


vqvae.load_state_dict(vqvae_state_dict_new)

del unet_state_dict, unet_state_dict_new, vqvae_state_dict, vqvae_state_dict_new

diffusion_pipeline = LDMPipeline(vqvae, unet, scheduler)
print("LOADED")


DTYPE = torch.float32
DEVICE = torch.device("cpu")
diffusion_pipeline.num_inference_steps = 20

diffusion_pipeline = diffusion_pipeline.to(device=DEVICE, dtype=DTYPE)
diffusion_pipeline.eval()


## Generate
noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)
start_time = time.time()
with torch.no_grad():
    for _ in range(10):
        sample = diffusion_pipeline(noise)

print(f"\n\nTime for 10 evals: {time.time() - start_time} s")

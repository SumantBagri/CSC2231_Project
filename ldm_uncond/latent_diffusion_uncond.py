import json
import re
import torch
import time

from collections import OrderedDict
from readers import BaseReader
from .model.unet import UNet2DModel
from .model.vq import VQModel
from .model.ddim_scheduler import DDIMScheduler


class LDMPipeline(torch.nn.Module):
	def __init__(self, reader=BaseReader()):
		super().__init__()

		# Init reader (for stat collection)
		self.reader = reader

		# Load model configurations
		unet_root = "ldm_uncond/ldm-celebahq-256/unet/"
		vqvae_root = "ldm_uncond/ldm-celebahq-256/vqvae/"
		scheduler_root = "ldm_uncond/ldm-celebahq-256/scheduler/"

		with open(unet_root + "config.json", 'r') as f:
			unet_config = json.load(f)
		with open(vqvae_root + "config.json", 'r') as f:
			vqvae_config = json.load(f)
		with open(scheduler_root + "scheduler_config.json", 'r') as f:
			scheduler_config = json.load(f)

		#----------------------------------------------------------------------------------------------------#
		#----------------------------------------------------------------------------------------------------#

		# Initalize model components from config
		self.unet = UNet2DModel(**unet_config)
		self.vqvae = VQModel(**vqvae_config)
		self.scheduler = DDIMScheduler(**scheduler_config)

		#----------------------------------------------------------------------------------------------------#
		#----------------------------------------------------------------------------------------------------#

		# Load state dictionary for UNet
		unet_state_dict = torch.load(unet_root + "diffusion_pytorch_model.bin", map_location='cpu')
		unet_state_dict_new = OrderedDict()

		for k, v in unet_state_dict.items():
			if k.startswith("mid_block.resnets.0"):
				name = k.replace('resnets.0', 'resnets_0')
			elif k.startswith("mid_block.resnets"):
				i = int(k[18])
				name = k[:18] + str(i-1) + k[19:]
			else:
				name = k
			unet_state_dict_new[name] = v

		self.unet.load_state_dict(unet_state_dict_new)

		#----------------------------------------------------------------------------------------------------#
		#----------------------------------------------------------------------------------------------------#

		# Load state dictionary for VQVAE
		vqvae_state_dict = torch.load(vqvae_root + "diffusion_pytorch_model.bin", map_location='cpu')
		vqvae_state_dict_new = OrderedDict()

		for k, v in vqvae_state_dict.items():
			if "mid_block.resnets.0" in k:
				name = k.replace('resnets.0', 'resnets_0')
			elif "mid_block.resnets" in k:
				name = re.sub(r"resnets.(\d+)", lambda match: "resnets." + str(int(match.group(1))-1), k)
			else:
				name = k
			vqvae_state_dict_new[name] = v

		self.vqvae.load_state_dict(vqvae_state_dict_new)

		#----------------------------------------------------------------------------------------------------#
		#----------------------------------------------------------------------------------------------------#

		# Delete state dictionaries to free up memory
		del unet_state_dict, unet_state_dict_new, vqvae_state_dict, vqvae_state_dict_new

		# Initialize default number of inference steps
		self.num_inference_steps = 20

	@torch.no_grad()
	def forward(self, noise):
		# set inference steps for DDIM
		self.scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)

		with torch.no_grad():
			# denoising loop
			image = noise
			stime = time.time()
			for t in self.scheduler.timesteps:
				print(f"Timestep = {t.item()}\r", end="")

				residual = self.unet(image, t.to(dtype=self.unet.dtype, device=self.unet.device))

				# compute previous image x_t according to DDIM formula
				prev_image = self.scheduler.step(residual, t, image, eta=0.0)[0]

				# x_t-1 -> x_t
				image = prev_image
			self.reader.metrics['lat'].maxvals[0] = round(time.time()-stime,4)
			
			# decode image with vae
			stime = time.time()
			image = self.vqvae.decode(image)
			self.reader.metrics['lat'].maxvals[1] = round(time.time()-stime,4)

			# process image
			image_processed = image.permute(0, 2, 3, 1)
			image_processed = (image_processed + 1.0) * 127.5
			image_processed = image_processed.clamp(0, 255)

			return image_processed
	
	def warmup(self, num_iter=3):
		print("Warming up...")
		tmp = self.num_inference_steps
		self.num_inference_steps = 2
		noise = torch.randn((1, 3, 64, 64), dtype=self.unet.dtype, device=self.unet.device)
		for _ in range(num_iter):
			self(noise)
		self.num_inference_steps = tmp
		print("Warmup complete!")
	
	@torch.inference_mode()
	@torch.autocast("cuda")
	def export_unet_to_onnx(self, fname="uldm_unet_fp16.onnx"):
		inputs = torch.randn((1, 3, 64, 64), dtype=self.unet.dtype, device=self.unet.device),\
				 torch.randn(1, dtype=self.unet.dtype, device=self.unet.device)
		
		print('Starting export to onnx...')
		
		# Export the model
		torch.onnx.export(self.unet,                 			  # model being run
						  inputs,                    			  # model input (or a tuple for multiple inputs)
						  fname,     			  # where to save the model (can be a file or file-like object)
						  export_params=True,        			  # store the trained parameter weights inside the model file
						  opset_version=13,          			  # the ONNX version to export the model to
						  do_constant_folding=True,  			  # whether to execute constant folding for optimization
						  verbose=False,             			  # set verbosity
						  input_names = ['input_0', 'input_1'],   # the model's input names
						  output_names = ['output_0'], 			  # the model's output names
                          dynamic_axes={'input_0' : {0 : 'batch_size'},    # variable length axes
                                        'input_1' : {0 : 'batch_size'},
										'output_0': {0 : 'batch_size'}}
						)

	@torch.inference_mode()
	@torch.autocast("cuda")
	def export_pipeline_to_onnx(self, fname="uldm_fp16.onnx"):
		inputs = torch.randn((1, 3, 64, 64), dtype=self.unet.dtype, device=self.unet.device)

		print('Starting export to onnx...')
		
		# Export the model
		torch.onnx.export(self,                 			      # model being run
						  inputs,                    			  # model input (or a tuple for multiple inputs)
						  fname,     			  # where to save the model (can be a file or file-like object)
						  export_params=True,        			  # store the trained parameter weights inside the model file
						  opset_version=13,          			  # the ONNX version to export the model to
						  do_constant_folding=True,  			  # whether to execute constant folding for optimization
						  verbose=False,             			  # set verbosity
						  input_names = ['input_0'],   # the model's input names
						  output_names = ['output_0'], 			  # the model's output names
                          dynamic_axes={'input_0' : {0 : 'batch_size'},    # variable length axes
										'output_0': {0 : 'batch_size'}}
						)

	def load_optimized_unet(self, fname, save_torch=False):
		DTYPE = self.unet.dtype
		DEVICE = self.unet.device
		
		# Load from TorchScript
		try:
			self.unet = torch.jit.load(fname, map_location=torch.device('cpu'))
			print("Loaded torchscript file successfully!")
		except Exception as e:
			print(f"Error while loading file: {e}")
		
		if save_torch:
			save_path = fname.split('.')[0]+".pt"
			print(f"Saving UNet to: {save_path}")
			unet_scripted = torch.jit.script(self.unet)
			unet_scripted.save(save_path)
		
		# Convert to original dtype and load to device
		self.unet.to(dtype=DTYPE, device=DEVICE)

		# Set dtype and device variables (used in forward() call)
		self.unet.dtype = DTYPE
		self.unet.device = DEVICE
		

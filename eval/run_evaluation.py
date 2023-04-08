#!/usr/bin/env python3

import argparse
import configparser
import os
import subprocess
import torch

from ldm_uncond.latent_diffusion_uncond import LDMPipeline
from evaluation import BaseEvaluator, ONNXEvaluator

##============================##
## Baseline - 32bit Model     ##
##============================##
def baseline(dev):
    print("\033[4mRunning Baseline\033[0m\n\n")

    # Init evaluator
    baseline_evaluator= BaseEvaluator(dev=dev, perf_cls="baseline")

    # Probe before model loading
    baseline_evaluator.reader.probe()

    # Init inputs and model
    DTYPE = torch.float32
    DEVICE = torch.device('cuda')
    diffusion_pipeline = LDMPipeline(reader=baseline_evaluator.reader).to(device=DEVICE, dtype=DTYPE)

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)

    # Evaluate the baseline
    baseline_evaluator.evaluate(diffusion_pipeline,noise)

    del diffusion_pipeline
    del baseline_evaluator
    torch.cuda.empty_cache()

##======================================##
## Optimization 1 - JIT Compilation     ##
##======================================##
def optim1(dev, infile):
    print("\n\n\033[4mRunning Optimization 1 - JIT Compilation\033[0m\n\n")

    # Init evaluator
    optim1_evaluator= BaseEvaluator(dev=dev, perf_cls="32bit_jit")

    # Probe before model loading
    optim1_evaluator.reader.probe()

    # Init inputs and model
    DTYPE = torch.float32
    DEVICE = torch.device('cuda')

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)
    torchscript_model = torch.jit.load(infile)

    # Evaluate optimization-1
    with torch.no_grad():
        optim1_evaluator.evaluate(torchscript_model,noise)

    del torchscript_model
    del optim1_evaluator
    torch.cuda.empty_cache()

##=============================================================##
## Optimization 2 - Quantization (16-bit) + JIT Compilation    ##
##=============================================================##
def optim2(dev, infile):
    print("\n\n\033[4mRunning Optimization 2 - Quantization (16-bit) + JIT Compilation\033[0m\n\n")

    # Init evaluator
    optim2_evaluator= BaseEvaluator(dev=dev, perf_cls="16bit_jit")

    # Probe before model loading
    optim2_evaluator.reader.probe()

    # Init inputs and model
    DTYPE = torch.float16
    DEVICE = torch.device('cuda')

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)
    torchscript_model = torch.jit.load(infile)

    # Evaluate optimization-2
    with torch.no_grad():
        optim2_evaluator.evaluate(torchscript_model,noise)

    del torchscript_model
    del optim2_evaluator
    torch.cuda.empty_cache()

##=============================================================================================##
## Optimization 3 - ONNX Runtime (Graph optimizations + Transformer specific optimizations)    ##
##=============================================================================================##
def optim3(dev, infile):
    print("\n\n\033[4mRunning Optimization 3 - ONNX Runtime (Graph optimizations + Transformer specific optimizations)\033[0m\n\n")

    # Init inputs and model
    DTYPE = torch.float32

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE)

    # #### Optimized ONNX
    # Init evaluator
    optim3_2 = ONNXEvaluator(infile, dev=dev, perf_cls="onnx_optim")
    with torch.no_grad():
        optim3_2.evaluate(noise)

    del optim3_2

##=============================================================================================##
## Optimization 4 - TensorRT (Layer & Tensor fusion + Quantization (16-bit) + JIT Compilation) ##
##=============================================================================================##
def optim4(dev, infile):
    print("\n\n\033[4mRunning Optimization 4 - TensorRT (Layer & Tensor fusion + Quantization (16-bit) + JIT Compilation)\033[0m\n\n")

    os.environ["CUDA_MODULE_LOADING"] = 'LAZY'

    # Init evaluator
    optim4_evaluator= BaseEvaluator(dev=dev, perf_cls="tensorRT")

    # Probe before model loading
    optim4_evaluator.reader.probe()

    # Init inputs and model
    DTYPE = torch.float16
    DEVICE = torch.device('cuda')

    # ### Load optimized UNet
    optimized_diffusion_pipeline = LDMPipeline(reader=optim4_evaluator.reader)

    optimized_diffusion_pipeline = optimized_diffusion_pipeline.to(device=DEVICE, dtype=DTYPE)
    optimized_diffusion_pipeline.load_optimized_unet(infile)

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)

    # Evaluate the baseline
    optim4_evaluator.evaluate(optimized_diffusion_pipeline,noise)

    del optimized_diffusion_pipeline
    del optim4_evaluator
    torch.cuda.empty_cache()

# Push metrics to github
def push(dev):
    try:
        os.system("git add output/eval_data/rtx_3070*")
        os.system(f"git commit -m 'Evaluation datafiles updated {dev}'")
        os.system("git push")
    except Exception as e:
        print(e)

# Get hardware device info
def get_device():
    # Check for RTX devices
    print("Checking for RTX devices...", end=' ')
    os.system('update-pciids > /dev/null 2>&1')
    try:
        lshw_out = subprocess.check_output('lshw', shell=True).decode('utf-8').replace("\n", "")
        if 'rtx 3070' in lshw_out.lower():
            print("\033[92mSuccess\033[0m\n")
            return 'rtx_3070'
        else:
            raise
    except:
        print("\033[91mFailed\033[0m\n")
    # Check for Jetson devices
    try:
        print("Checking for Jetson devices...", end=' ')
        if os.environ.get('DEVICE').lower() == "jetson_nano":
            print("\033[92mSuccess\033[0m\n")
            return 'jetson_nano'
        else:
            raise
    except:
        print("\033[91mFailed\033[0m\n")
        pass
    # No other devices supported (yet)
    return


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Runs evaluation pipeline')
    parser.add_argument('-c', '--config', type=str, help='Path to config file')
    args = parser.parse_args()

    # Read config file
    config = configparser.ConfigParser()
    config.read(args.config)

    # Parse arguments from config file
    baseline = config.getboolean('arguments', 'baseline')
    o1_file = config.get('arguments', 'optim1', fallback=None)
    o2_file = config.get('arguments', 'optim2', fallback=None)
    o3_file = config.get('arguments', 'optim3', fallback=None)
    o4_file = config.get('arguments', 'optim4', fallback=None)
    all_functions = config.getboolean('arguments', 'all')
    git_push = config.getboolean('arguments', 'git_push')

    if not os.environ.get('CUDA_LAUNCH_BLOCKING'):
        print("Inference script should be run as:")
        print("\t$ CUDA_LAUNCH_BLOCKING=1 ./run_evaluation.py [OPTIONS]")
        exit(1)

    dev = get_device()
    if not dev:
        print("No capable devices found!")
        exit(1)

    print("========================================================")
    print(f"\033[4mRunning Evaluations for device: {dev}\033[0m")
    print("========================================================")

    # Run pipeline based on arguments
    if baseline:
        baseline(dev)
    if o1_file:
        optim1(dev, o1_file)
    if o2_file:
        optim2(dev, o2_file)
    if o3_file:
        optim3(dev, o3_file)
    if o4_file:
        optim4(dev, o4_file)
    if all_functions:
        baseline()
        optim1(dev, o1_file)
        optim2(dev, o2_file)
        optim3(dev, o3_file)
        optim4(dev, o4_file)        
    if git_push:
        push(dev)
    
    exit(0)


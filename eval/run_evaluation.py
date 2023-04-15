#!/usr/bin/env python3

import argparse
import toml
import os
import subprocess

os.environ["CUDA_MODULE_LOADING"] = 'LAZY'

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

##=============================================##
## Optimization 1 - Quantization (16-bit)      ##
##=============================================##
def optim1(dev):
    print("\033[4mRunning Optimization 1 - Quantization (16-bit)\033[0m\n\n")

    # Init evaluator
    optim1_evaluator= BaseEvaluator(dev=dev, perf_cls="fp16")

    # Probe before model loading
    optim1_evaluator.reader.probe()

    # Init inputs and model
    DTYPE = torch.float16
    DEVICE = torch.device('cuda')
    diffusion_pipeline = LDMPipeline(reader=optim1_evaluator.reader).to(device=DEVICE, dtype=DTYPE)

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)

    # Evaluate optimization-1
    optim1_evaluator.evaluate(diffusion_pipeline,noise)

    del diffusion_pipeline
    del optim1_evaluator
    torch.cuda.empty_cache()

##=============================================##
## Optimization 2 - JIT Compilation (32-bit)   ##
##=============================================##
def optim2(dev):
    print("\n\n\033[4mRunning Optimization 2 - JIT Compilation (32-bit)\033[0m\n\n")

    # Init evaluator
    optim2_evaluator= BaseEvaluator(dev=dev, perf_cls="fp32_jit")

    # Probe before model loading
    optim2_evaluator.reader.probe()

    # Init inputs and model
    DTYPE = torch.float32
    DEVICE = torch.device('cuda')

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)
    # diffusion_pipeline = LDMPipeline(reader=optim2_evaluator.reader).to(device=DEVICE, dtype=DTYPE)
    # diffusion_pipeline.eval()
    # print("Performing JIT trace...")
    # with torch.no_grad():
    #     torchscript_model = torch.jit.trace(diffusion_pipeline, noise)
    torchscript_model = torch.jit.load('output/optim/uldm_jit_fp32.ptl')
    torchscript_model.to(dtype=DTYPE, device=DEVICE)
    
    # Evaluate optimization-2
    with torch.no_grad():
        optim2_evaluator.evaluate(torchscript_model,noise)

    del torchscript_model
    del optim2_evaluator
    torch.cuda.empty_cache()

##=============================================================##
## Optimization 3 - Quantization (16-bit) + JIT Compilation    ##
##=============================================================##
def optim3(dev):
    print("\n\n\033[4mRunning Optimization 3 - Quantization (16-bit) + JIT Compilation\033[0m\n\n")

    # Init evaluator
    optim3_evaluator= BaseEvaluator(dev=dev, perf_cls="fp16_jit")

    # Probe before model loading
    optim3_evaluator.reader.probe()

    # Init inputs and model
    DTYPE = torch.float16
    DEVICE = torch.device('cuda')

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)
    # diffusion_pipeline = LDMPipeline(reader=optim3_evaluator.reader).to(device=DEVICE, dtype=DTYPE)
    # diffusion_pipeline.eval()
    # print("Performing JIT trace...")
    # with torch.no_grad():
    #     torchscript_model = torch.jit.trace(diffusion_pipeline, noise)
    torchscript_model = torch.jit.load('output/optim/uldm_jit_fp16.ptl')
    torchscript_model.to(dtype=DTYPE, device=DEVICE)

    # Evaluate optimization-3
    with torch.no_grad():
        optim3_evaluator.evaluate(torchscript_model,noise)

    del torchscript_model
    del optim3_evaluator
    torch.cuda.empty_cache()

##=============================================================================================##
## Optimization 4 - ONNX Runtime (Graph optimizations + Transformer specific optimizations)    ##
##=============================================================================================##
def optim4(dev, infile, providers=["CPUExecutionProvider"]):
    print("\n\n\033[4mRunning Optimization 4 - ONNX Runtime (Graph optimizations + Transformer specific optimizations)\033[0m\n\n")

    # Init inputs and model
    DTYPE = torch.float32

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE)

    # #### Optimized ONNX
    # Init evaluator
    optim4_evaluator = ONNXEvaluator(infile, dev=dev, perf_cls="onnx_optim", providers=providers)
    with torch.no_grad():
        optim4_evaluator.evaluate(noise)

    del optim4_evaluator

##=============================================================================================##
## Optimization 5 - TensorRT (Layer & Tensor fusion + Quantization (16-bit) + JIT Compilation) ##
##=============================================================================================##
def optim5(dev, infile):
    print("\n\n\033[4mRunning Optimization 4 - TensorRT (Layer & Tensor fusion + Quantization (16-bit) + JIT Compilation)\033[0m\n\n")

    # Init evaluator
    optim5_evaluator= BaseEvaluator(dev=dev, perf_cls="tensorRT")

    # Probe before model loading
    optim5_evaluator.reader.probe()

    # Init inputs and model
    DTYPE = torch.float16
    DEVICE = torch.device('cuda')

    # ### Load optimized UNet
    optimized_diffusion_pipeline = LDMPipeline(reader=optim5_evaluator.reader)

    optimized_diffusion_pipeline = optimized_diffusion_pipeline.to(device=DEVICE, dtype=DTYPE)
    optimized_diffusion_pipeline.load_optimized_unet(infile)

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)

    # Evaluate the baseline
    optim5_evaluator.evaluate(optimized_diffusion_pipeline,noise)

    del optimized_diffusion_pipeline
    del optim5_evaluator
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

    if not os.environ.get('CUDA_LAUNCH_BLOCKING'):
        print("\nCUDA launch environment varible not defined!")
        print("Inference script should be run from the project root directory as:")
        print("\t$ CUDA_LAUNCH_BLOCKING=1 ./eval/run_evaluation.py -c eval/config.toml")
        exit(1)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Runs evaluation pipeline')
    parser.add_argument('-c', '--config', type=str, help='Path to config file')
    args = parser.parse_args()

    # Read config file
    config = toml.load(args.config)

    # Print the config file to stdout
    print("========================================================")
    print("LOADED CONFIGURATION FILE:")
    print("========================================================")
    print(toml.dumps(config))
    print("========================================================")

    user_decision = input("Proceed with evaluations? (Y/n)")
    if user_decision.lower() == 'n':
        exit(0)

    # Parse arguments from config file
    run_baseline = config['arguments'][0]['baseline']
    run_o1 = config['arguments'][0]['optim1']
    run_o2 = config['arguments'][0]['optim2']
    run_o3 = config['arguments'][0]['optim3']
    o4_file = config['arguments'][0]['optim4']
    o5_file = config['arguments'][0]['optim5']
    all_functions = config['arguments'][0]['all']
    git_push = config['arguments'][0]['git_push']

    # Parse the provider information
    providers = [(p.pop('name'),p) for p in config['providers']]


    dev = get_device()
    if not dev:
        print("No capable devices found!")
        exit(1)

    print("========================================================")
    print(f"\033[4mRunning Evaluations for device: {dev}\033[0m")
    print("========================================================")

    # Run pipeline based on arguments
    if run_baseline:
        baseline(dev)
    if run_o1:
        optim1(dev)
    if run_o2:
        optim2(dev)
    if run_o3:
        optim3(dev)
    if o4_file:
        optim4(dev, o4_file, providers)
    if o5_file:
        optim5(dev, o5_file)
    if all_functions:
        baseline()
        optim1(dev)
        optim2(dev)
        optim3(dev)
        optim4(dev, o4_file, providers)
        optim5(dev, o5_file)     
    if git_push:
        push(dev)
    
    exit(0)


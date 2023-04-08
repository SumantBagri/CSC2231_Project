#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
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
def optim1(dev):
    print("\n\n\033[4mRunning Optimization 1 - JIT Compilation\033[0m\n\n")

    # Init evaluator
    optim1_evaluator= BaseEvaluator(dev=dev, perf_cls="32bit_jit")

    # Probe before model loading
    optim1_evaluator.reader.probe()

    # Init inputs and model
    DTYPE = torch.float32
    DEVICE = torch.device('cuda')

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)
    torchscript_model = torch.jit.load("output/optim/model_jit_fp32_cuda.ptl")

    # Evaluate optimization-1
    with torch.no_grad():
        optim1_evaluator.evaluate(torchscript_model,noise)

    del torchscript_model
    del optim1_evaluator
    torch.cuda.empty_cache()

##=============================================================##
## Optimization 2 - Quantization (16-bit) + JIT Compilation    ##
##=============================================================##
def optim2(dev):
    print("\n\n\033[4mRunning Optimization 2 - Quantization (16-bit) + JIT Compilation\033[0m\n\n")

    # Init evaluator
    optim2_evaluator= BaseEvaluator(dev=dev, perf_cls="16bit_jit")

    # Probe before model loading
    optim2_evaluator.reader.probe()

    # Init inputs and model
    DTYPE = torch.float16
    DEVICE = torch.device('cuda')

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)
    torchscript_model = torch.jit.load("output/optim/model_jit_fp16_cuda.ptl")

    # Evaluate optimization-2
    with torch.no_grad():
        optim2_evaluator.evaluate(torchscript_model,noise)

    del torchscript_model
    del optim2_evaluator
    torch.cuda.empty_cache()

##=============================================================================================##
## Optimization 3 - ONNX Runtime (Graph optimizations + Transformer specific optimizations)    ##
##=============================================================================================##
def optim3(dev):
    print("\n\n\033[4mRunning Optimization 3 - ONNX Runtime (Graph optimizations + Transformer specific optimizations)\033[0m\n\n")

    # Init inputs and model
    DTYPE = torch.float32

    noise = torch.randn((1, 3, 64, 64), dtype=DTYPE)

    # #### Vanilla ONNX
    print("\tVanilla ONNX")

    # Init evaluator
    optim3_1 = ONNXEvaluator("output/optim/model_onnx_fp32_cpu.onnx", dev=dev, perf_cls="onnx_vanilla")
    optim3_1.evaluate(noise)

    del optim3_1

    # #### Optimized ONNX
    print("\tOptimized ONNX")

    # Init evaluator
    optim3_2 = ONNXEvaluator("output/optim/model_onnx_fp32_cpu_optimized.onnx", dev=dev, perf_cls="onnx_optim")
    optim3_2.evaluate(noise)

    del optim3_2

    # #### Transformer Optimized ONNX
    print("\tTransformer-Optimized ONNX")
    
    # Init evaluator
    optim3_3 = ONNXEvaluator("output/optim/model_onnx_fp32_cpu_optimized_tf.onnx", dev=dev, perf_cls="onnx_optim_tf")
    optim3_3.evaluate(noise)

    del optim3_3

##=============================================================================================##
## Optimization 4 - TensorRT (Layer & Tensor fusion + Quantization (16-bit) + JIT Compilation) ##
##=============================================================================================##
def optim4(dev):
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
    optimized_diffusion_pipeline.load_optimized_unet("output/optim/uldm_unet_fp16_sim.ts")

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

def get_device():
    # Check for RTX devices
    print("Checking for RTX devices...", end=' ')
    os.system('update-pciids > /dev/null 2>&1')
    lshw_out = subprocess.check_output('/usr/bin/lshw', shell=True).decode('utf-8').replace("\n", "")
    if 'rtx 3070' in lshw_out.lower():
        print("\033[92mSuccess\033[0m")
        return 'rtx_3070'
    else:
        print("\033[91mFailed\033[0m")
    # Check for Jetson devices
    print("Checking for Jetson devices...", end=' ')
    if 'jetson nano' in os.environ['DEVICE']:
        print("\033[92mSuccess\033[0m")
        return 'jetson_nano'
    else:
        print("\033[91mFailed\033[0m")
    # No other devices supported (yet)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs evaluation pipeline')
    parser.add_argument('-b', '--baseline', action='store_true', help='Run baseline function')
    parser.add_argument('-o1', '--optim1', action='store_true', help='Run optim1 function')
    parser.add_argument('-o2', '--optim2', action='store_true', help='Run optim2 function')
    parser.add_argument('-o3', '--optim3', action='store_true', help='Run optim3 function')
    parser.add_argument('-o4', '--optim4', action='store_true', help='Run optim4 function')
    parser.add_argument('-a', '--all', action='store_true', help='Run all functions')
    parser.add_argument('-gp', '--git-push', action='store_true', help="Push all the output files to github")
    args = parser.parse_args()

    if not os.environ.get('CUDA_LAUNCH_BLOCKING'):
        print("Inference script should be run as:")
        print("\t$ CUDA_LAUNCH_BLOCKING=1 ./run_evaluation.py [OPTIONS]")
        exit(1)

    dev = get_device()
    if not dev:
        print("No capable devices found!")
        exit(1)

    print("========================================================")
    print(f"\n\n\033[4mRunning Evaluations for device: {dev}\033[0m")
    print("========================================================")

    if args.baseline:
        baseline(dev)
    if args.optim1 and dev == 'rtx_3070':
        optim1()
    if args.optim2 and dev == 'rtx_3070':
        optim2()
    if args.optim3 and dev == 'rtx_3070':
        optim3()
    if args.optim4 and dev == 'rtx_3070':
        optim4()
    if args.all:
        baseline()
        if dev == 'rtx_3070':
            optim1()
            optim2()
            optim3()
            optim4()
    if args.git_push:
        push()
    
    exit(0)


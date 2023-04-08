#!/usr/bin/env python3

import argparse
import os
import torch

from ldm_uncond.latent_diffusion_uncond import LDMPipeline
from evaluation import BaseEvaluator, ONNXEvaluator

# Define hardware
dev = "rtx_3070"


##============================##
## Baseline - 32bit Model     ##
##============================##
def baseline():
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
def optim1():
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
def optim2():
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
def optim3():
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
def optim4():
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
def push():
    try:
        os.system("git add output/eval_data/rtx_3070*")
        os.system("git commit -m 'Evaluation datafiles updated (RTX 3070)'")
        os.system("git push")
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run evaluation pipeline on RTX 3070')
    parser.add_argument('-b', '--baseline', action='store_true', help='Run baseline function')
    parser.add_argument('-o1', '--optim1', action='store_true', help='Run optim1 function')
    parser.add_argument('-o2', '--optim2', action='store_true', help='Run optim2 function')
    parser.add_argument('-o3', '--optim3', action='store_true', help='Run optim3 function')
    parser.add_argument('-o4', '--optim4', action='store_true', help='Run optim4 function')
    parser.add_argument('-a', '--all', action='store_true', help='Run all functions')
    parser.add_argument('-gp', '--git-push', action='store_true', help="Push all the output files to github")
    args = parser.parse_args()

    print(args)
    exit

    if args.baseline:
        baseline()
    if args.optim1:
        optim1()
    if args.optim2:
        optim2()
    if args.optim3:
        optim3()
    if args.optim4:
        optim4()
    if args.all:
        baseline()
        optim1()
        optim2()
        optim3()
        optim4()
    if args.git_push:
        push()


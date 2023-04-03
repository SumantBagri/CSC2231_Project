import torch
import torch.onnx

from ldm_uncond.latent_diffusion_uncond import LDMPipeline

if __name__ == '__main__':
    diffusion_pipeline = LDMPipeline()

    DTYPE = torch.float16
    DEVICE = torch.device('cuda')

    diffusion_pipeline = diffusion_pipeline.to(device=DEVICE, dtype=DTYPE)

    with torch.inference_mode(), torch.autocast("cuda"):
        print('Starting export to onnx...')
        dummy_input = torch.randn((1, 3, 64, 64), dtype=DTYPE, device=DEVICE)
        # Export the model
        torch.onnx.export(diffusion_pipeline,        # model being run
                          dummy_input,               # model input (or a tuple for multiple inputs)
                          "diffusion_model1.onnx",    # where to save the model (can be a file or file-like object)
                          export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=11,          # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          verbose=False,             # set verbosity
                          input_names = ['input'],   # the model's input names
                          output_names = ['output'], # the model's output names
                          dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})
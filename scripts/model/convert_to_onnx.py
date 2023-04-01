import torch
import torch.onnx

from ldm_uncond.latent_diffusion_uncond import LDMPipeline

if __name__ == '__main__':
    diffusion_pipeline = LDMPipeline()

    DTYPE = torch.float16
    DEVICE = torch.device('cuda')
    diffusion_pipeline.num_inference_steps = 20
    BATCH_SIZE = 1

    diffusion_pipeline = diffusion_pipeline.to(device=DEVICE, dtype=DTYPE)
    diffusion_pipeline.eval()
    dummy_input = torch.randn((BATCH_SIZE, 3, 64, 64), dtype=DTYPE, device=DEVICE)

    print('Starting export to onnx...')
    # Export the model
    torch.onnx.export(diffusion_pipeline,        # model being run
                      dummy_input,               # model input (or a tuple for multiple inputs)
                      "diffusion_model.onnx",    # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=False,             # set verbosity
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
import tensorrt as trt
import numpy as np

# Load the TensorRT engine
engine_file = 'resnet50_pytorch.trt'
with open(engine_file, 'rb') as f:
    engine_data = f.read()
runtime = trt.Runtime(trt.Logger())
engine = runtime.deserialize_cuda_engine(engine_data)

# Get the network definition
network = engine.get_network()

# Count the number of FLOPs
flops = 0
for layer in network:
    for index in range(layer.num_outputs):
        tensor = layer.get_output(index)
        shape = tensor.shape
        if len(shape) == 0:
            # Skip scalar tensors
            continue
        size = np.prod(shape)
        flops += size * tensor.dtype.itemsize * 2

print(f"Number of FLOPs: {flops}")

# Can't seem to get the network from the engine. however, the code is specific to the model rather than the device anyway. we can probably skip flops or manually do this calculation if time permits

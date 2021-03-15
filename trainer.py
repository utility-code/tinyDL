import numpy as np

def linear(input_dim, output_dim):
    return {"input_dim":input_dim, "output_dim": output_dim}

def initLayers(arch, seed = 99):
    np.random.seed(seed)
    num_layers = len(arch)
    param_values = {}

    for idx, layer in enumerate(arch):
        current_idx = idx+1
        layer_in_size = layer["input_dim"]
        layer_out_size = layer["output_dim"]

        param_values['W'+ str(current_idx)] = np.random.randn(
            layer_out_size , layer_in_size
        ) * 0.1
        param_values['b'+ str(current_idx)] = np.random.randn(
            layer_out_size , 1
        ) * 0.1

    return param_values

arch = [linear(2, 25), linear(25, 50)]
print(initLayers(arch))

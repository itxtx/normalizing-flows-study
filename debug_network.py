import torch
import numpy as np
from src.flows.advanced.neural_autoregressive_flow import DeepMADE

# Test network structure
input_dim = 2
hidden_dims = [4]

made = DeepMADE(input_dim=input_dim, hidden_dims=hidden_dims)

print("Network structure:")
for name, module in made.network.named_modules():
    if hasattr(module, 'weight'):
        print(f"{name}: {module}")
        if hasattr(module, 'mask'):
            print(f"  Mask shape: {module.mask.shape}")
            print(f"  Mask:\n{module.mask}")
        print(f"  Weight shape: {module.weight.shape}")
        print(f"  Weight:\n{module.weight}")
        print()

# Test step by step computation
x = torch.tensor([[1.0, 0.0], [1.0, 2.0]])
print(f"Input: {x}")

# Forward through each layer
current = x
for i, layer in enumerate(made.network):
    if hasattr(layer, 'forward'):
        current = layer(current)
        print(f"After layer {i} ({type(layer).__name__}): {current}")

print(f"Final output: {current}")
mu, alpha = current.chunk(2, dim=1)
print(f"mu: {mu}")
print(f"alpha: {alpha}")
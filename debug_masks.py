import torch
import numpy as np
from src.flows.advanced.neural_autoregressive_flow import DeepMADE

# Test mask creation
input_dim = 2
hidden_dims = [4]

made = DeepMADE(input_dim=input_dim, hidden_dims=hidden_dims)

print("Input dim:", input_dim)
print("Hidden dims:", hidden_dims)
print("Degrees:")
for key, value in made.m.items():
    print(f"  Layer {key}: {value}")

print("\nMasks:")
for i, mask in enumerate(made.masks):
    print(f"  Mask {i} shape: {mask.shape}")
    print(f"  Mask {i}:\n{mask}")

# Test with a simple input
x = torch.tensor([[1.0, 0.0], [1.0, 2.0]])
output = made(x)
print(f"\nInput: {x}")
print(f"Output: {output}")

mu, alpha = output.chunk(2, dim=1)
print(f"mu: {mu}")
print(f"alpha: {alpha}")

# Check if first dimension is independent of second dimension
print(f"\nFirst dimension independence test:")
print(f"mu[0,0] == mu[1,0]: {torch.allclose(mu[0,0], mu[1,0])}")
print(f"alpha[0,0] == alpha[1,0]: {torch.allclose(alpha[0,0], alpha[1,0])}")
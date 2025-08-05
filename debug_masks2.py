import torch
import numpy as np
from src.flows.advanced.neural_autoregressive_flow import DeepMADE

# Test mask creation with detailed analysis
input_dim = 2
hidden_dims = [4]

made = DeepMADE(input_dim=input_dim, hidden_dims=hidden_dims)

print("Input dim:", input_dim)
print("Hidden dims:", hidden_dims)
print("Degrees:")
for key, value in made.m.items():
    print(f"  Layer {key}: {value}")

print("\nMask analysis:")
for i, mask in enumerate(made.masks):
    print(f"  Mask {i} shape: {mask.shape}")
    print(f"  Mask {i}:\n{mask}")

# Analyze the final mask creation logic
print("\nFinal mask creation analysis:")
output_dim = input_dim * 2
m_final = np.zeros((output_dim, hidden_dims[-1]), dtype=np.float32)
for i in range(input_dim):
    for k in range(2):  # output_dim_multiplier = 2
        out_idx = i * 2 + k
        condition = made.m[len(hidden_dims) - 1] <= i - 1
        print(f"Output dim {i}, param {k} (idx {out_idx}): hidden degrees {made.m[len(hidden_dims) - 1]} <= {i-1} = {condition}")
        m_final[out_idx, :] = condition.astype(np.float32)

print(f"Expected final mask:\n{m_final}")

# Test what should happen:
# - Output 0 (mu_0, alpha_0) should not depend on any input -> mask should be all zeros
# - Output 1 (mu_1, alpha_1) should depend on input 0 -> mask should connect to hidden units with degree 0

print("\nCorrect autoregressive structure:")
print("- Output 0 (mu_0, alpha_0) should be independent of all inputs")
print("- Output 1 (mu_1, alpha_1) should depend only on input 0")
print("- Hidden units with degree 0 can see input 0")
print("- Hidden units with degree 1 can see inputs 0 and 1")
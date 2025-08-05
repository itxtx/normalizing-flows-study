import torch
from src.flows.advanced.neural_autoregressive_flow import NeuralAutoregressiveFlow

# Simple test
dim = 2
naf = NeuralAutoregressiveFlow(dim=dim, hidden_dims=[8])

# Test with simple input
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("Original x:", x)

z, log_det_inv = naf.inverse(x)
print("z from inverse:", z)
print("log_det_inv:", log_det_inv)

x_reconstructed, log_det_fwd = naf.forward(z)
print("Reconstructed x:", x_reconstructed)
print("log_det_fwd:", log_det_fwd)

print("Difference:", torch.abs(x - x_reconstructed))
print("Max difference:", torch.max(torch.abs(x - x_reconstructed)))

# Check if the conditioner is working properly
print("\nTesting conditioner directly:")
params = naf.conditioner(x)
mu, alpha = params.chunk(2, dim=1)
print("mu:", mu)
print("alpha:", alpha)

# Test autoregressive property
x_test = torch.tensor([[1.0, 0.0], [1.0, 2.0]])
params1 = naf.conditioner(x_test)
mu1, alpha1 = params1.chunk(2, dim=1)
print("\nAutoregressive test:")
print("x_test:", x_test)
print("mu1:", mu1)
print("alpha1:", alpha1)

# The first dimension output should be the same for both inputs
print("mu1[0,0] == mu1[1,0]:", torch.allclose(mu1[0,0], mu1[1,0]))
print("alpha1[0,0] == alpha1[1,0]:", torch.allclose(alpha1[0,0], alpha1[1,0]))
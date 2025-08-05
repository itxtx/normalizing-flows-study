import torch
from src.flows.autoregressive.masked_autoregressive_flow import MaskedAutoregressiveFlow

# Test existing MAF for comparison
dim = 4
batch_size = 32

maf = MaskedAutoregressiveFlow(dim=dim, hidden_dim=64)

# Test forward-inverse consistency
x = torch.randn(batch_size, dim)
z, log_det_inv = maf.inverse(x)
x_reconstructed, log_det_fwd = maf.forward(z)

print("MAF Forward-Inverse Consistency Test:")
print("Max difference:", torch.max(torch.abs(x - x_reconstructed)))
print("Log-det consistency:", torch.allclose(log_det_inv, -log_det_fwd, atol=1e-3))

# Test with smaller example
dim = 2
batch_size = 4
maf_small = MaskedAutoregressiveFlow(dim=dim, hidden_dim=8)

x_small = torch.randn(batch_size, dim)
z_small, log_det_inv_small = maf_small.inverse(x_small)
x_reconstructed_small, log_det_fwd_small = maf_small.forward(z_small)

print("\nMAF Small Test:")
print("Original x:", x_small)
print("Reconstructed x:", x_reconstructed_small)
print("Max difference:", torch.max(torch.abs(x_small - x_reconstructed_small)))
print("Log-det inv:", log_det_inv_small)
print("Log-det fwd:", log_det_fwd_small)
print("Log-det consistency:", torch.allclose(log_det_inv_small, -log_det_fwd_small, atol=1e-3))
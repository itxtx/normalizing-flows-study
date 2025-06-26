import torch
from sklearn.datasets import make_moons
import numpy as np

def get_two_moons_data(n_samples=10000, noise=0.05):
    """
    Generates and returns the two-moons dataset as a PyTorch tensor.

    Args:
        n_samples (int): The total number of points to generate.
        noise (float): Standard deviation of Gaussian noise added to the data.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 2) containing the data.
    """
    # Generate data using scikit-learn
    moons, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    
    # Convert to PyTorch tensor
    return torch.from_numpy(moons.astype(np.float32))


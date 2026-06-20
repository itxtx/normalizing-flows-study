import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    """A linear layer with a fixed binary mask."""
    
    def __init__(self, in_features, out_features, mask, bias=True):
        # Call the parent constructor
        super().__init__(in_features, out_features, bias)
        
        # Register the mask as a buffer to ensure it moves with the model (e.g., to GPU)
        self.register_buffer('mask', mask)

    def forward(self, input):
        # Apply the mask to the weights during the forward pass
        # Ensure mask has the same dtype as weights
        mask = self.mask.to(dtype=self.weight.dtype)
        return F.linear(input, self.weight * mask, self.bias)

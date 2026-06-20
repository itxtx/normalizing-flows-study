import torch
import torch.nn as nn
import torch.nn.functional as F
from src.flows.advanced.transformer_block import TransformerBlock

class CausalTransformer(nn.Module):
    """Causal Transformer for autoregressive flow conditioning."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim))  # Max sequence length
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim // num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim * 2)  # mu and alpha
        
        # Causal mask
        self.register_buffer('causal_mask', self._create_causal_mask(1000))
        
    def _create_causal_mask(self, seq_len):
        """Create causal mask for autoregressive property."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()
    
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        if seq_len <= self.pos_embedding.size(1):
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            # Extend positional encoding if needed
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2), 
                size=seq_len, 
                mode='linear'
            ).transpose(1, 2)
        
        x = x + pos_emb
        
        # Apply causal mask
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Project to output
        output = self.output_proj(x)
        
        return output

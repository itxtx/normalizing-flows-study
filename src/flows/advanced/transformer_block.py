import torch.nn as nn
from src.flows.advanced.multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    """Transformer block with causal attention and feed-forward network."""
    
    def __init__(self, dim, num_heads=8, head_dim=64, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads, head_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x), mask)
        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

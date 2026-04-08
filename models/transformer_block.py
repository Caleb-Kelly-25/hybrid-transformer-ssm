import torch
import torch.nn as nn
from typing import Optional
import math

class TransformerBlock(nn.Module):
    """Standard Transformer block with Pre-LN architecture."""
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.1,
        use_rope: bool = True,
        ff_mult: int = 4,
        causal: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        # Multi-head attention
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (Pre-LN)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.causal = causal
        self.use_rope = use_rope
        if use_rope:
            from positional_encoding import RotaryPositionalEncoding
            self.rope = RotaryPositionalEncoding(self.head_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN for attention
        normed = self.norm1(x)
        
        # Self-attention
        B, L, D = normed.shape
        qkv = self.qkv(normed).reshape(B, L, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # (B, L, heads, head_dim)
        
        if self.use_rope:
            q = self.rope(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            k = self.rope(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('b l h d, b s h d -> b h l s', q, k) * scale
        
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float('-inf'))
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('b h l s, b s h d -> b l h d', attn, v)
        out = out.reshape(B, L, D)
        out = self.proj(out)
        
        x = x + self.dropout(out)
        
        # Feed-forward
        x = x + self.ff(self.norm2(x))
        
        return x
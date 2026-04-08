import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 16384, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, d_model: int, max_len: int = 16384):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Precompute frequencies
        freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('freqs', freqs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device)
        freqs = torch.outer(t, self.freqs)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Apply rotation
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rotated * sin

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings."""
    def __init__(self, d_model: int, max_len: int = 16384):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]
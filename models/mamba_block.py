import torch
import torch.nn as nn
from typing import Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Using fallback implementation.")


class MambaBlock(nn.Module):
    """Mamba SSM block with selective state space."""
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        
        if MAMBA_AVAILABLE:
            self.norm = nn.LayerNorm(dim)
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # Fallback to simplified SSM for development
            self.norm = nn.LayerNorm(dim)
            self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
            self.activation = nn.SiLU()
            self.linear = nn.Linear(dim, dim * expand)
            self.linear2 = nn.Linear(dim * expand, dim)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        if MAMBA_AVAILABLE:
            x = self.mamba(x)
        else:
            # Simplified SSM implementation
            x_conv = x.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :x.size(1)]
            x_conv = x_conv.transpose(1, 2)
            x = self.activation(x_conv)
            
            x = self.linear(x)
            x = self.activation(x)
            x = self.linear2(x)
            
        x = self.dropout(x)
        return residual + x
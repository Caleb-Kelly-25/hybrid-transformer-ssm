import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal


class AdaptiveScheduler(nn.Module):
    """
    Learnable adaptive scheduler for routing between Transformer and SSM.
    Supports token-level and sequence-level routing with soft/hard decisions.
    """
    
    def __init__(
        self,
        dim: int,
        temperature: float = 1.0,
        hard: bool = False,
        token_level: bool = True,
        num_experts: int = 2,  # Transformer, SSM
        use_length_embedding: bool = True,
        max_seq_len: int = 16384
    ):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.hard = hard
        self.token_level = token_level
        self.num_experts = num_experts
        self.use_length_embedding = use_length_embedding
        
        # Router network
        router_dim = dim
        if use_length_embedding:
            self.length_embedding = nn.Embedding(max_seq_len, 32)
            router_dim += 32
            
        self.router = nn.Sequential(
            nn.Linear(router_dim, router_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(router_dim // 2, num_experts)
        )
        
        # For sequence-level routing
        if not token_level:
            self.pool_proj = nn.Linear(dim, dim)
            
        # Learnable temperature (optional)
        self.log_temp = nn.Parameter(torch.tensor(temperature).log())
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, L, D) input features
            lengths: (B,) sequence lengths
            return_logits: if True, return logits before softmax
            
        Returns:
            weights: (B, L, 2) if token_level else (B, 2)
            logits: if return_logits
        """
        B, L, D = x.shape
        
        if self.token_level:
            # Token-level routing
            router_input = x
            
            if self.use_length_embedding and lengths is not None:
                len_emb = self.length_embedding(lengths)  # (B, 32)
                len_emb = len_emb.unsqueeze(1).expand(-1, L, -1)
                router_input = torch.cat([router_input, len_emb], dim=-1)
                
            logits = self.router(router_input)  # (B, L, num_experts)
        else:
            # Sequence-level routing
            pooled = x.mean(dim=1)
            pooled = self.pool_proj(pooled)
            
            if self.use_length_embedding and lengths is not None:
                len_emb = self.length_embedding(lengths)
                pooled = torch.cat([pooled, len_emb], dim=-1)
                
            logits = self.router(pooled)  # (B, num_experts)
            logits = logits.unsqueeze(1).expand(-1, L, -1)  # (B, L, num_experts)
        
        # Get temperature
        temperature = self.log_temp.exp()
        
        if self.hard:
            # Hard routing with Gumbel-Softmax
            weights = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        else:
            # Soft routing
            weights = F.softmax(logits / temperature, dim=-1)
        
        if return_logits:
            return weights, logits
        return weights
    
    def get_routing_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of routing distribution."""
        return -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
    
    def get_routing_sparsity(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute sparsity (how confident the routing is)."""
        max_prob = weights.max(dim=-1)[0]
        return (max_prob > 0.9).float().mean()
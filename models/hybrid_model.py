import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Literal
import math

from transformer_block import TransformerBlock
from mamba_block import MambaBlock
from scheduler import AdaptiveScheduler
from positional_encoding import (
    PositionalEncoding,
    RotaryPositionalEncoding,
    LearnablePositionalEncoding
)


class HybridTransformerSSM(nn.Module):
    """
    Hybrid model with adaptive routing between Transformer and Mamba SSM blocks.
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        num_classes: int = 10,
        dropout: float = 0.1,
        max_seq_len: int = 16384,
        # Scheduler settings
        scheduler_temperature: float = 1.0,
        scheduler_hard: bool = False,
        scheduler_token_level: bool = True,
        # Positional encoding
        pos_encoding: Literal['sinusoidal', 'rope', 'learnable', 'none'] = 'sinusoidal',
        # Architecture choices
        use_pre_norm: bool = True,
        use_residual: bool = True,
        # Efficiency
        checkpoint_activations: bool = False,
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.num_classes = num_classes
        self.checkpoint_activations = checkpoint_activations
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding
        self.pos_encoding_type = pos_encoding
        if pos_encoding == 'sinusoidal':
            self.pos_encoding = PositionalEncoding(dim, max_seq_len, dropout)
        elif pos_encoding == 'rope':
            self.pos_encoding = RotaryPositionalEncoding(dim, max_seq_len)
        elif pos_encoding == 'learnable':
            self.pos_encoding = LearnablePositionalEncoding(dim, max_seq_len)
        else:
            self.pos_encoding = None
            
        # Create layers with schedulers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer_dict = nn.ModuleDict({
                'transformer': TransformerBlock(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    use_rope=(pos_encoding == 'rope')
                ),
                'mamba': MambaBlock(
                    dim=dim,
                    dropout=dropout
                ),
                'scheduler': AdaptiveScheduler(
                    dim=dim,
                    temperature=scheduler_temperature,
                    hard=scheduler_hard,
                    token_level=scheduler_token_level,
                    use_length_embedding=True,
                    max_seq_len=max_seq_len
                )
            })
            self.layers.append(layer_dict)
            
        # Final layer norm
        self.final_norm = nn.LayerNorm(dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        return_routing: bool = False,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hybrid model.
        
        Args:
            input_ids: (B, L) token indices
            attention_mask: (B, L) attention mask
            lengths: (B,) sequence lengths
            return_routing: return routing weights for analysis
            return_hidden_states: return all intermediate hidden states
            
        Returns:
            Dictionary containing:
                - logits: (B, num_classes)
                - routing_weights: list of (B, L, 2) if return_routing
                - hidden_states: list of (B, L, D) if return_hidden_states
        """
        B, L = input_ids.shape
        
        # Get lengths if not provided
        if lengths is None and attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
        elif lengths is None:
            lengths = torch.full((B,), L, device=input_ids.device)
            
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        if self.pos_encoding is not None:
            if self.pos_encoding_type == 'rope':
                # RoPE is applied inside attention blocks
                pass
            else:
                x = self.pos_encoding(x)
                
        x = self.dropout(x)
        
        # Store outputs
        routing_weights_list = []
        hidden_states_list = []
        
        if return_hidden_states:
            hidden_states_list.append(x)
            
        # Process through layers
        for layer_idx, layer in enumerate(self.layers):
            # Get routing weights
            scheduler = layer['scheduler']
            weights, logits = scheduler(x, lengths=lengths, return_logits=True)
            routing_weights_list.append(weights.detach())
            
            # Apply transformer and mamba
            if self.checkpoint_activations and self.training:
                # Gradient checkpointing for memory efficiency
                t_out = torch.utils.checkpoint.checkpoint(
                    layer['transformer'], x, attention_mask, use_reentrant=False
                )
                m_out = torch.utils.checkpoint.checkpoint(
                    layer['mamba'], x, use_reentrant=False
                )
            else:
                t_out = layer['transformer'](x, mask=attention_mask)
                m_out = layer['mamba'](x)
            
            # Weighted combination
            w_t = weights[..., 0:1]  # (B, L, 1)
            w_m = weights[..., 1:2]  # (B, L, 1)
            
            x = w_t * t_out + w_m * m_out
            
            if return_hidden_states:
                hidden_states_list.append(x)
                
        # Final normalization
        x = self.final_norm(x)
        
        # Pooling (mean pooling over sequence)
        if attention_mask is not None:
            # Masked mean pooling
            x = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)
            
        # Classification
        logits = self.head(x)
        
        output = {'logits': logits}
        
        if return_routing:
            output['routing_weights'] = routing_weights_list
            
        if return_hidden_states:
            output['hidden_states'] = hidden_states_list
            
        return output
    
    def get_routing_statistics(self) -> Dict[str, float]:
        """Get statistics about routing behavior."""
        stats = {}
        for layer_idx, layer in enumerate(self.layers):
            scheduler = layer['scheduler']
            stats[f'layer_{layer_idx}_temperature'] = scheduler.log_temp.exp().item()
            
        return stats
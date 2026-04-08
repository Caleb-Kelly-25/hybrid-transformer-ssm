import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class HybridLoss(nn.Module):
    """
    Combined loss for hybrid model training.
    Includes task loss, routing entropy, and compute-aware penalties.
    """
    
    def __init__(
        self,
        task_weight: float = 1.0,
        entropy_weight: float = 0.01,
        compute_weight: float = 0.01,
        balance_weight: float = 0.01,
        target_transformer_ratio: float = 0.3
    ):
        super().__init__()
        self.task_weight = task_weight
        self.entropy_weight = entropy_weight
        self.compute_weight = compute_weight
        self.balance_weight = balance_weight
        self.target_transformer_ratio = target_transformer_ratio
        
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        routing_weights: List[torch.Tensor],
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: (B, num_classes) model predictions
            labels: (B,) ground truth labels
            routing_weights: List of (B, L, 2) routing weights per layer
            return_components: if True, return dict of individual losses
        """
        
        # Task loss
        task_loss = self.ce_loss(logits, labels)
        
        # Routing entropy loss (encourage confident routing)
        entropy_loss = 0.0
        for weights in routing_weights:
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            entropy_loss += entropy
        entropy_loss /= len(routing_weights)
        
        # Compute-aware loss (penalize transformer usage)
        compute_loss = 0.0
        for weights in routing_weights:
            transformer_usage = weights[..., 0].mean()
            compute_loss += transformer_usage
        compute_loss /= len(routing_weights)
        
        # Balance loss (keep transformer usage near target ratio)
        balance_loss = 0.0
        for weights in routing_weights:
            actual_ratio = weights[..., 0].mean()
            balance_loss += (actual_ratio - self.target_transformer_ratio) ** 2
        balance_loss /= len(routing_weights)
        
        # Combined loss
        total_loss = (
            self.task_weight * task_loss +
            self.entropy_weight * entropy_loss +
            self.compute_weight * compute_loss +
            self.balance_weight * balance_loss
        )
        
        if return_components:
            return total_loss, {
                'task_loss': task_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'compute_loss': compute_loss.item(),
                'balance_loss': balance_loss.item(),
                'total_loss': total_loss.item()
            }
        
        return total_loss


class MemoryEfficientLoss(HybridLoss):
    """Loss function with gradient checkpointing support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward_with_checkpoint(self, model, batch):
        """Forward pass with gradient checkpointing."""
        def custom_forward(input_ids, attention_mask, lengths):
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lengths=lengths,
                return_routing=True
            )
        
        outputs = torch.utils.checkpoint.checkpoint(
            custom_forward,
            batch['input_ids'],
            batch['attention_mask'],
            batch['lengths'],
            use_reentrant=False
        )
        
        return self.forward(
            outputs['logits'],
            batch['labels'],
            outputs['routing_weights'],
            return_components=True
        )
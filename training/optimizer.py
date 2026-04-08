"""
Custom optimizers and learning rate schedulers for hybrid model training.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW, Adam
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Optional, List, Dict, Any


class HybridOptimizer:
    """
    Wrapper for optimizer with special handling for routing parameters.
    Allows different learning rates for routing vs main parameters.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 3e-4,
        routing_lr_multiplier: float = 1.0,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8
    ):
        self.model = model
        self.base_lr = base_lr
        self.routing_lr = base_lr * routing_lr_multiplier
        
        # Separate parameters
        routing_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'scheduler' in name or 'router' in name:
                routing_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {
                'params': other_params,
                'lr': base_lr,
                'weight_decay': weight_decay
            },
            {
                'params': routing_params,
                'lr': self.routing_lr,
                'weight_decay': weight_decay * 0.1  # Less regularization for routing
            }
        ]
        
        self.optimizer = AdamW(
            param_groups,
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        
        print(f"HybridOptimizer created:")
        print(f"  - Main params: {sum(p.numel() for p in other_params):,} (LR: {base_lr})")
        print(f"  - Routing params: {sum(p.numel() for p in routing_params):,} (LR: {self.routing_lr})")
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            progress = step / self.warmup_steps
            return [base_lr * progress for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and linear decay.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            progress = step / self.warmup_steps
            return [base_lr * progress for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = max(0.0, 1.0 - progress)
            
            return [
                self.min_lr + (base_lr - self.min_lr) * progress
                for base_lr in self.base_lrs
            ]


class CyclicCosineScheduler(_LRScheduler):
    """
    Cyclic cosine learning rate scheduler with restarts.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        cycle_steps: int,
        cycle_mult: float = 2.0,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.cycle_steps = cycle_steps
        self.cycle_mult = cycle_mult
        self.min_lr = min_lr
        self.cycles_completed = 0
        self.steps_in_cycle = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        # Check if we've completed a cycle
        if step > 0 and step % self.cycle_steps == 0:
            self.cycles_completed += 1
            self.cycle_steps = int(self.cycle_steps * self.cycle_mult)
        
        self.steps_in_cycle = step % self.cycle_steps
        
        # Cosine cycle
        progress = self.steps_in_cycle / self.cycle_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]


class LayerwiseLR:
    """
    Layer-wise learning rate decay for transformer models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        decay_rate: float = 0.95,
        min_lr_ratio: float = 0.1
    ):
        self.model = model
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        self.min_lr = base_lr * min_lr_ratio
        
    def create_param_groups(self) -> List[Dict[str, Any]]:
        """Create parameter groups with layer-wise learning rates."""
        param_groups = []
        
        # Get number of layers
        num_layers = len(self.model.layers)
        
        for layer_idx, layer in enumerate(self.model.layers):
            # Deeper layers get lower learning rates
            depth_ratio = layer_idx / num_layers
            lr_mult = self.decay_rate ** (num_layers - layer_idx)
            lr = max(self.base_lr * lr_mult, self.min_lr)
            
            param_groups.append({
                'params': layer.parameters(),
                'lr': lr,
                'name': f'layer_{layer_idx}',
                'depth_ratio': depth_ratio
            })
        
        # Embedding and head get base learning rate
        other_params = []
        for name, param in self.model.named_parameters():
            if not any(f'layers.{i}' in name for i in range(num_layers)):
                other_params.append(param)
        
        param_groups.append({
            'params': other_params,
            'lr': self.base_lr,
            'name': 'embeddings_and_head'
        })
        
        return param_groups


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    Improves generalization by finding flatter minima.
    """
    
    def __init__(
        self,
        params,
        base_optimizer: Optimizer = AdamW,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """First step of SAM: compute gradients and take a step towards sharp region."""
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # Climb to local maximum
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Second step of SAM: update weights from the sharp region."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # Recover original parameters
        
        self.base_optimizer.step()  # Take optimizer step
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """Combined first and second steps."""
        assert closure is not None, "SAM requires closure for first step"
        
        closure = torch.enable_grad()(closure)
        
        # First step
        self.first_step(zero_grad=True)
        closure()
        
        # Second step
        self.second_step()
    
    def _grad_norm(self):
        """Compute gradient norm."""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict[str, Any]
) -> tuple:
    """
    Factory function to create optimizer and scheduler from config.
    """
    optimizer_type = config.get('optimizer', 'adamw')
    
    # Create optimizer
    if optimizer_type == 'adamw':
        if config.get('separate_routing_lr', False):
            optimizer = HybridOptimizer(
                model,
                base_lr=config['learning_rate'],
                routing_lr_multiplier=config.get('routing_lr_multiplier', 1.0),
                weight_decay=config.get('weight_decay', 0.01)
            ).optimizer
        else:
            optimizer = AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 0.01),
                betas=(0.9, 0.999)
            )
    
    elif optimizer_type == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
    
    elif optimizer_type == 'sam':
        optimizer = SAM(
            model.parameters(),
            base_optimizer=AdamW,
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01),
            rho=config.get('sam_rho', 0.05),
            adaptive=config.get('sam_adaptive', False)
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Create scheduler
    scheduler_type = config.get('scheduler', 'cosine')
    total_steps = config['epochs'] * config.get('steps_per_epoch', 1000)
    warmup_steps = config.get('warmup_steps', int(total_steps * 0.1))
    
    if scheduler_type == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'linear':
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'cyclic':
        scheduler = CyclicCosineScheduler(
            optimizer,
            cycle_steps=config.get('cycle_steps', total_steps // 3),
            min_lr=config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return optimizer, scheduler


if __name__ == "__main__":
    # Test optimizers
    from models.hybrid_model import HybridTransformerSSM
    
    model = HybridTransformerSSM(
        vocab_size=1000,
        dim=256,
        depth=6,
        heads=8,
        num_classes=10
    )
    
    config = {
        'optimizer': 'adamw',
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'separate_routing_lr': True,
        'routing_lr_multiplier': 2.0,
        'scheduler': 'cosine',
        'warmup_steps': 1000,
        'epochs': 50,
        'steps_per_epoch': 100
    }
    
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    print(f"Optimizer: {optimizer}")
    print(f"Scheduler: {scheduler}")
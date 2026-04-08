import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from typing import Dict, Optional, Any
import time
import os

from models.hybrid_model import HybridTransformerSSM
from losses import HybridLoss
from metrics import compute_metrics
from utils.memory_utils import MemoryTracker


class HybridTrainer:
    """
    Trainer for Hybrid Transformer-SSM model with advanced features.
    """
    
    def __init__(
        self,
        model: HybridTransformerSSM,
        config: Dict[str, Any],
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Training configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        self.criterion = HybridLoss(
            task_weight=config.get('task_weight', 1.0),
            entropy_weight=config.get('entropy_weight', 0.01),
            compute_weight=config.get('compute_weight', 0.01),
            balance_weight=config.get('balance_weight', 0.01),
            target_transformer_ratio=config.get('target_transformer_ratio', 0.3)
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        warmup_steps = config.get('warmup_steps', 1000)
        total_steps = config['epochs'] * len(train_loader)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        self.use_amp = config.get('use_amp', True)
        
        # Memory tracking
        self.memory_tracker = MemoryTracker()
        
        # Logging
        self.use_wandb = config.get('use_wandb', True)
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb_project', 'hybrid-transformer-ssm'),
                config=config,
                name=config.get('experiment_name', None)
            )
            
        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Metrics tracking
        self.best_val_accuracy = 0.0
        self.global_step = 0
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_metrics = {}
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    lengths=batch['lengths'],
                    return_routing=True
                )
                
                loss, loss_components = self.criterion(
                    outputs['logits'],
                    batch['labels'],
                    outputs['routing_weights'],
                    return_components=True
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Compute metrics
            with torch.no_grad():
                metrics = compute_metrics(
                    outputs['logits'],
                    batch['labels'],
                    outputs['routing_weights']
                )
            
            # Update totals
            total_loss += loss_components['total_loss']
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_components['total_loss']:.4f}",
                'acc': f"{metrics['accuracy']:.3f}",
                't_ratio': f"{metrics['transformer_ratio']:.2f}"
            })
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train/loss': loss_components['total_loss'],
                    'train/accuracy': metrics['accuracy'],
                    'train/transformer_ratio': metrics['transformer_ratio'],
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/step': self.global_step,
                    **{f'train/{k}': v for k, v in loss_components.items()}
                })
                
            self.global_step += 1
            
            # Memory tracking
            if batch_idx % 100 == 0:
                mem_stats = self.memory_tracker.get_stats()
                if self.use_wandb:
                    wandb.log({f'memory/{k}': v for k, v in mem_stats.items()})
        
        # Average metrics
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate(self, loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate model on given dataloader."""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        all_routing_weights = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc='Validation'):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    lengths=batch['lengths'],
                    return_routing=True
                )
                
                loss, _ = self.criterion(
                    outputs['logits'],
                    batch['labels'],
                    outputs['routing_weights'],
                    return_components=True
                )
                
                total_loss += loss.item()
                all_logits.append(outputs['logits'])
                all_labels.append(batch['labels'])
                all_routing_weights.extend(outputs['routing_weights'])
        
        # Compute overall metrics
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        metrics = compute_metrics(all_logits, all_labels, all_routing_weights)
        metrics['loss'] = total_loss / len(loader)
        
        return metrics
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        for epoch in range(self.config['epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if self.val_loader:
                val_metrics = self.validate(self.val_loader)
                
                # Log validation metrics
                if self.use_wandb:
                    wandb.log({
                        'val/loss': val_metrics['loss'],
                        'val/accuracy': val_metrics['accuracy'],
                        'val/transformer_ratio': val_metrics['transformer_ratio'],
                        'epoch': epoch
                    })
                
                # Save best model
                if val_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['accuracy']
                    self.save_checkpoint('best_model.pt')
                    
                print(f"Epoch {epoch}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.3f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.3f}")
            else:
                print(f"Epoch {epoch}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.3f}")
                
            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_freq', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # Final evaluation on test set
        if self.test_loader:
            test_metrics = self.validate(self.test_loader)
            print(f"\nTest Results:")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.4f}")
                
            if self.use_wandb:
                wandb.log({f'test/{k}': v for k, v in test_metrics.items()})
        
        if self.use_wandb:
            wandb.finish()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'global_step': self.global_step,
            'config': self.config
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded from {path}")
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import pandas as pd


class RoutingVisualizer:
    """Visualize routing behavior of hybrid model."""
    
    def __init__(self, save_dir: str = './figures'):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_routing_heatmap(
        self,
        routing_weights: List[torch.Tensor],
        input_ids: torch.Tensor,
        save_name: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot heatmap of routing decisions across layers and tokens.
        
        Args:
            routing_weights: List of (B, L, 2) tensors, one per layer
            input_ids: (B, L) input tokens
            save_name: filename to save figure
        """
        num_layers = len(routing_weights)
        batch_idx = 0  # Visualize first example in batch
        
        # Extract transformer weights
        transformer_weights = []
        for layer_weights in routing_weights:
            # (B, L, 2) -> (L,)
            t_weights = layer_weights[batch_idx, :, 0].cpu().numpy()
            transformer_weights.append(t_weights)
        
        # Create heatmap
        transformer_weights = np.array(transformer_weights)  # (num_layers, L)
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, 
                                 gridspec_kw={'height_ratios': [3, 1]})
        
        # Routing heatmap
        im = axes[0].imshow(transformer_weights, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        axes[0].set_ylabel('Layer')
        axes[0].set_xlabel('Token Position')
        axes[0].set_title('Transformer Routing Probability\n(Green=Transformer, Red=Mamba)')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[0])
        
        # Token values
        tokens = input_ids[batch_idx, :].cpu().numpy()
        axes[1].plot(tokens, 'b-', alpha=0.7, linewidth=0.5)
        axes[1].set_ylabel('Token ID')
        axes[1].set_xlabel('Token Position')
        axes[1].set_title('Input Tokens')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_layer_routing_distribution(
        self,
        routing_weights: List[torch.Tensor],
        save_name: Optional[str] = None
    ):
        """
        Plot distribution of routing decisions per layer.
        """
        num_layers = len(routing_weights)
        
        # Compute statistics
        layer_stats = []
        for layer_idx, weights in enumerate(routing_weights):
            t_ratio = weights[..., 0].mean().item()
            t_std = weights[..., 0].std().item()
            layer_stats.append({
                'layer': layer_idx,
                'transformer_ratio': t_ratio,
                'mamba_ratio': 1 - t_ratio,
                'std': t_std
            })
        
        df = pd.DataFrame(layer_stats)
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        x = np.arange(num_layers)
        width = 0.35
        
        axes[0].bar(x, df['transformer_ratio'], width, label='Transformer', color='#2ecc71')
        axes[0].bar(x, df['mamba_ratio'], width, bottom=df['transformer_ratio'], 
                   label='Mamba', color='#e74c3c')
        
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Routing Ratio')
        axes[0].set_title('Layer-wise Routing Distribution')
        axes[0].set_xticks(x)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Line plot with std
        axes[1].errorbar(x, df['transformer_ratio'], yerr=df['std'], 
                        fmt='o-', capsize=5, color='#3498db', linewidth=2)
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Transformer Ratio')
        axes[1].set_title('Transformer Usage by Layer (with std)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_sequence_length_analysis(
        self,
        length_ratios: List[Tuple[int, float]],
        save_name: Optional[str] = None
    ):
        """
        Plot how routing changes with sequence length.
        
        Args:
            length_ratios: List of (sequence_length, transformer_ratio) tuples
        """
        lengths, ratios = zip(*length_ratios)
        
        plt.figure(figsize=(10, 6))
        plt.plot(lengths, ratios, 'o-', linewidth=2, markersize=8, color='#2ecc71')
        plt.xlabel('Sequence Length')
        plt.ylabel('Transformer Ratio')
        plt.title('Transformer Usage vs Sequence Length')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Add trend line
        z = np.polyfit(np.log(lengths), ratios, 1)
        p = np.poly1d(z)
        plt.plot(lengths, p(np.log(lengths)), '--', alpha=0.5, 
                label=f'Trend: {z[0]:.3f} log(length) + {z[1]:.3f}')
        plt.legend()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_training_curves(
        self,
        metrics_history: Dict[str, List[float]],
        save_name: Optional[str] = None
    ):
        """
        Plot training curves for multiple metrics.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss
        axes[0, 0].plot(metrics_history['train_loss'], label='Train', linewidth=2)
        if 'val_loss' in metrics_history:
            axes[0, 0].plot(metrics_history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(metrics_history['train_accuracy'], label='Train', linewidth=2)
        if 'val_accuracy' in metrics_history:
            axes[0, 1].plot(metrics_history['val_accuracy'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Transformer ratio
        axes[0, 2].plot(metrics_history['transformer_ratio'], linewidth=2, color='#e67e22')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Transformer Ratio')
        axes[0, 2].set_title('Transformer Usage Over Time')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Routing entropy
        axes[1, 0].plot(metrics_history['routing_entropy'], linewidth=2, color='#9b59b6')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].set_title('Routing Entropy Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(metrics_history['learning_rate'], linewidth=2, color='#1abc9c')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        # F1 Score
        axes[1, 2].plot(metrics_history['train_f1'], label='Train', linewidth=2)
        if 'val_f1' in metrics_history:
            axes[1, 2].plot(metrics_history['val_f1'], label='Val', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1 Score')
        axes[1, 2].set_title('Training and Validation F1')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        plt.show()
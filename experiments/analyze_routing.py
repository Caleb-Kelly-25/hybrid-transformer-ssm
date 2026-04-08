"""
Analyze routing behavior of trained hybrid models.
Provides detailed statistics and visualizations of routing decisions.
"""

import torch
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('.')

from models.hybrid_model import HybridTransformerSSM
from data.dataloaders import create_dataloader
from evaluation.visualization import RoutingVisualizer


class RoutingAnalyzer:
    """Comprehensive routing behavior analysis."""
    
    def __init__(
        self,
        model: HybridTransformerSSM,
        device: torch.device,
        save_dir: str = './routing_analysis'
    ):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        
    def analyze_layer_specialization(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Analyze how different layers specialize.
        
        Returns:
            Dictionary with layer-wise statistics
        """
        print("Analyzing layer specialization...")
        
        # Accumulators
        layer_weights = defaultdict(list)
        layer_entropies = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                if batch_idx >= num_batches:
                    break
                    
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    lengths=batch['lengths'],
                    return_routing=True
                )
                
                # Collect per-layer statistics
                for layer_idx, weights in enumerate(outputs['routing_weights']):
                    # Transformer ratio
                    t_ratio = weights[..., 0].mean().item()
                    layer_weights[layer_idx].append(t_ratio)
                    
                    # Entropy
                    entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item()
                    layer_entropies[layer_idx].append(entropy)
        
        # Compute statistics
        stats = {
            'layer_transformer_ratio_mean': np.array([
                np.mean(layer_weights[i]) for i in range(len(layer_weights))
            ]),
            'layer_transformer_ratio_std': np.array([
                np.std(layer_weights[i]) for i in range(len(layer_weights))
            ]),
            'layer_entropy_mean': np.array([
                np.mean(layer_entropies[i]) for i in range(len(layer_entropies))
            ]),
            'layer_entropy_std': np.array([
                np.std(layer_entropies[i]) for i in range(len(layer_entropies))
            ])
        }
        
        # Save to file
        np.savez(self.save_dir / 'layer_specialization.npz', **stats)
        
        # Create visualization
        self._plot_layer_specialization(stats)
        
        return stats
    
    def analyze_token_position_effects(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Analyze how routing changes with token position.
        """
        print("Analyzing token position effects...")
        
        all_position_weights = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                if batch_idx >= num_batches:
                    break
                    
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    lengths=batch['lengths'],
                    return_routing=True
                )
                
                # Average across layers
                stacked_weights = torch.stack(outputs['routing_weights'])  # (num_layers, B, L, 2)
                avg_weights = stacked_weights.mean(dim=0)  # (B, L, 2)
                
                # Collect position-wise weights
                for b in range(avg_weights.shape[0]):
                    seq_len = batch['lengths'][b].item()
                    position_weights = avg_weights[b, :seq_len, 0].cpu().numpy()
                    
                    for pos, weight in enumerate(position_weights):
                        all_position_weights.append({
                            'position': pos,
                            'normalized_position': pos / seq_len,
                            'transformer_weight': weight,
                            'sequence_length': seq_len
                        })
        
        # Convert to DataFrame
        df = pd.DataFrame(all_position_weights)
        
        # Compute statistics by position percentile
        df['position_percentile'] = pd.cut(
            df['normalized_position'],
            bins=10,
            labels=[f"{i*10}-{(i+1)*10}%" for i in range(10)]
        )
        
        position_stats = df.groupby('position_percentile').agg({
            'transformer_weight': ['mean', 'std', 'count']
        }).reset_index()
        
        # Save to CSV
        df.to_csv(self.save_dir / 'position_effects.csv', index=False)
        position_stats.to_csv(self.save_dir / 'position_stats.csv')
        
        # Create visualization
        self._plot_position_effects(df)
        
        return {'raw_data': df, 'stats': position_stats}
    
    def analyze_length_scaling(
        self,
        dataloader: torch.utils.data.DataLoader,
        length_buckets: List[int] = [512, 1024, 2048, 4096, 8192, 16384],
        num_batches_per_bucket: int = 5
    ) -> Dict[str, List]:
        """
        Analyze how routing changes with sequence length.
        """
        print("Analyzing length scaling...")
        
        bucket_stats = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    lengths=batch['lengths'],
                    return_routing=True
                )
                
                # Process each sequence individually
                for b in range(batch['input_ids'].shape[0]):
                    seq_len = batch['lengths'][b].item()
                    
                    # Find appropriate bucket
                    bucket = min(length_buckets, key=lambda x: abs(x - seq_len))
                    
                    # Get average transformer ratio across layers
                    seq_weights = []
                    for layer_weights in outputs['routing_weights']:
                        t_ratio = layer_weights[b, :seq_len, 0].mean().item()
                        seq_weights.append(t_ratio)
                    
                    avg_transformer_ratio = np.mean(seq_weights)
                    
                    bucket_stats[bucket].append({
                        'length': seq_len,
                        'transformer_ratio': avg_transformer_ratio,
                        'layer_weights': seq_weights
                    })
        
        # Compute statistics per bucket
        length_stats = {}
        for bucket, data in bucket_stats.items():
            ratios = [d['transformer_ratio'] for d in data]
            length_stats[bucket] = {
                'mean_ratio': np.mean(ratios),
                'std_ratio': np.std(ratios),
                'num_samples': len(ratios),
                'avg_length': np.mean([d['length'] for d in data])
            }
        
        # Save results
        with open(self.save_dir / 'length_scaling.json', 'w') as f:
            json.dump(length_stats, f, indent=2)
        
        # Create visualization
        self._plot_length_scaling(length_stats)
        
        return length_stats
    
    def analyze_token_type_preferences(
        self,
        dataloader: torch.utils.data.DataLoader,
        vocab_size: int = 1000,
        num_batches: int = 10
    ) -> pd.DataFrame:
        """
        Analyze which token types prefer Transformer vs Mamba.
        """
        print("Analyzing token type preferences...")
        
        token_weights = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                if batch_idx >= num_batches:
                    break
                    
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    lengths=batch['lengths'],
                    return_routing=True
                )
                
                # Average across layers
                stacked_weights = torch.stack(outputs['routing_weights'])
                avg_weights = stacked_weights.mean(dim=0)  # (B, L, 2)
                
                # Collect per-token preferences
                for b in range(batch['input_ids'].shape[0]):
                    seq_len = batch['lengths'][b].item()
                    tokens = batch['input_ids'][b, :seq_len].cpu().numpy()
                    weights = avg_weights[b, :seq_len, 0].cpu().numpy()
                    
                    for token, weight in zip(tokens, weights):
                        token_weights[int(token)].append(weight)
        
        # Compute statistics per token
        token_stats = []
        for token_id in range(vocab_size):
            if token_weights[token_id]:
                token_stats.append({
                    'token_id': token_id,
                    'mean_transformer_ratio': np.mean(token_weights[token_id]),
                    'std_transformer_ratio': np.std(token_weights[token_id]),
                    'count': len(token_weights[token_id])
                })
        
        df = pd.DataFrame(token_stats)
        df = df.sort_values('mean_transformer_ratio', ascending=False)
        
        # Save to CSV
        df.to_csv(self.save_dir / 'token_preferences.csv', index=False)
        
        # Create visualization
        self._plot_token_preferences(df)
        
        return df
    
    def analyze_routing_consistency(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_passes: int = 5,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Analyze consistency of routing decisions with dropout enabled.
        """
        print("Analyzing routing consistency...")
        
        self.model.train()  # Enable dropout for variability
        
        all_consistencies = []
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * dataloader.batch_size >= num_samples:
                break
                
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Multiple forward passes
            all_weights = []
            for _ in range(num_passes):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    lengths=batch['lengths'],
                    return_routing=True
                )
                all_weights.append(torch.stack(outputs['routing_weights']))
            
            # Stack all passes: (num_passes, num_layers, B, L, 2)
            all_weights = torch.stack(all_weights)
            
            # Compute pairwise consistency
            for i in range(num_passes):
                for j in range(i + 1, num_passes):
                    # Hard decision consistency
                    hard_i = (all_weights[i, ..., 0] > 0.5).float()
                    hard_j = (all_weights[j, ..., 0] > 0.5).float()
                    consistency = (hard_i == hard_j).float().mean().item()
                    all_consistencies.append(consistency)
        
        self.model.eval()
        
        stats = {
            'mean_consistency': np.mean(all_consistencies),
            'std_consistency': np.std(all_consistencies),
            'min_consistency': np.min(all_consistencies),
            'max_consistency': np.max(all_consistencies)
        }
        
        # Save results
        with open(self.save_dir / 'routing_consistency.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def generate_comprehensive_report(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict:
        """
        Generate a comprehensive routing analysis report.
        """
        print("\n" + "="*50)
        print("GENERATING COMPREHENSIVE ROUTING REPORT")
        print("="*50 + "\n")
        
        report = {}
        
        # 1. Layer specialization
        print("\n1. Layer Specialization Analysis")
        print("-" * 40)
        layer_stats = self.analyze_layer_specialization(dataloader)
        report['layer_specialization'] = layer_stats
        
        print(f"Layer transformer ratios:")
        for i, (mean, std) in enumerate(zip(
            layer_stats['layer_transformer_ratio_mean'],
            layer_stats['layer_transformer_ratio_std']
        )):
            print(f"  Layer {i}: {mean:.3f} ± {std:.3f}")
        
        # 2. Position effects
        print("\n2. Token Position Effects")
        print("-" * 40)
        position_stats = self.analyze_token_position_effects(dataloader)
        report['position_effects'] = position_stats
        
        # 3. Length scaling
        print("\n3. Sequence Length Scaling")
        print("-" * 40)
        length_stats = self.analyze_length_scaling(dataloader)
        report['length_scaling'] = length_stats
        
        print("Length vs Transformer ratio:")
        for length, stats in sorted(length_stats.items()):
            print(f"  ~{length}: {stats['mean_ratio']:.3f} ± {stats['std_ratio']:.3f}")
        
        # 4. Token preferences
        print("\n4. Token Type Preferences")
        print("-" * 40)
        token_df = self.analyze_token_type_preferences(dataloader)
        report['token_preferences'] = token_df
        
        print("Top 5 Transformer-preferring tokens:")
        for _, row in token_df.head().iterrows():
            print(f"  Token {int(row['token_id'])}: {row['mean_transformer_ratio']:.3f}")
        
        print("\nTop 5 Mamba-preferring tokens:")
        for _, row in token_df.tail().iterrows():
            print(f"  Token {int(row['token_id'])}: {row['mean_transformer_ratio']:.3f}")
        
        # 5. Consistency
        print("\n5. Routing Consistency")
        print("-" * 40)
        consistency_stats = self.analyze_routing_consistency(dataloader)
        report['consistency'] = consistency_stats
        
        print(f"Mean consistency: {consistency_stats['mean_consistency']:.3f}")
        print(f"Std consistency: {consistency_stats['std_consistency']:.3f}")
        
        # Save complete report
        with open(self.save_dir / 'comprehensive_report.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_report = {}
            for key, value in report.items():
                if isinstance(value, dict):
                    serializable_report[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                elif isinstance(value, pd.DataFrame):
                    serializable_report[key] = value.to_dict('records')
                else:
                    serializable_report[key] = value
            
            json.dump(serializable_report, f, indent=2)
        
        print("\n" + "="*50)
        print(f"Report saved to {self.save_dir}/")
        print("="*50)
        
        return report
    
    def _plot_layer_specialization(self, stats: Dict[str, np.ndarray]):
        """Plot layer specialization statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        num_layers = len(stats['layer_transformer_ratio_mean'])
        x = np.arange(num_layers)
        
        # Transformer ratio by layer
        axes[0].errorbar(
            x,
            stats['layer_transformer_ratio_mean'],
            yerr=stats['layer_transformer_ratio_std'],
            fmt='o-',
            capsize=5,
            linewidth=2,
            markersize=8,
            color='#2ecc71'
        )
        axes[0].set_xlabel('Layer Index')
        axes[0].set_ylabel('Transformer Ratio')
        axes[0].set_title('Transformer Usage by Layer')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(x)
        
        # Add horizontal line at 0.5
        axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Routing entropy by layer
        axes[1].errorbar(
            x,
            stats['layer_entropy_mean'],
            yerr=stats['layer_entropy_std'],
            fmt='o-',
            capsize=5,
            linewidth=2,
            markersize=8,
            color='#e74c3c'
        )
        axes[1].set_xlabel('Layer Index')
        axes[1].set_ylabel('Routing Entropy')
        axes[1].set_title('Routing Uncertainty by Layer')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(x)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'layer_specialization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_position_effects(self, df: pd.DataFrame):
        """Plot position effects on routing."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Average by absolute position (first 100 positions)
        pos_data = df[df['position'] < 100].groupby('position')['transformer_weight'].agg(['mean', 'std'])
        
        axes[0].plot(pos_data.index, pos_data['mean'], 'b-', linewidth=2, label='Mean')
        axes[0].fill_between(
            pos_data.index,
            pos_data['mean'] - pos_data['std'],
            pos_data['mean'] + pos_data['std'],
            alpha=0.3,
            color='blue'
        )
        axes[0].set_xlabel('Token Position')
        axes[0].set_ylabel('Transformer Weight')
        axes[0].set_title('Routing vs Absolute Position (First 100 tokens)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # By normalized position percentile
        percentiles = df.groupby('position_percentile')['transformer_weight'].agg(['mean', 'std'])
        
        axes[1].bar(
            range(len(percentiles)),
            percentiles['mean'],
            yerr=percentiles['std'],
            capsize=5,
            color='#3498db',
            alpha=0.7
        )
        axes[1].set_xlabel('Position Percentile')
        axes[1].set_ylabel('Transformer Weight')
        axes[1].set_title('Routing by Normalized Position')
        axes[1].set_xticks(range(len(percentiles)))
        axes[1].set_xticklabels(percentiles.index, rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'position_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_length_scaling(self, length_stats: Dict):
        """Plot length scaling analysis."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        lengths = []
        means = []
        stds = []
        
        for length, stats in sorted(length_stats.items()):
            lengths.append(length)
            means.append(stats['mean_ratio'])
            stds.append(stats['std_ratio'])
        
        ax.errorbar(
            lengths,
            means,
            yerr=stds,
            fmt='o-',
            capsize=5,
            linewidth=2,
            markersize=10,
            color='#e67e22'
        )
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Transformer Ratio')
        ax.set_title('Transformer Usage vs Sequence Length')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(lengths) > 1:
            z = np.polyfit(np.log(lengths), means, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(lengths), max(lengths), 100)
            ax.plot(x_trend, p(np.log(x_trend)), '--', 
                   label=f'Trend: {z[0]:.3f} log(L) + {z[1]:.3f}',
                   alpha=0.7, color='gray')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'length_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_token_preferences(self, df: pd.DataFrame):
        """Plot token type preferences."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Top and bottom 20 tokens
        top_20 = df.head(20)
        bottom_20 = df.tail(20)
        
        # Plot top Transformer-preferring
        axes[0].barh(range(20), top_20['mean_transformer_ratio'].values, 
                    xerr=top_20['std_transformer_ratio'].values,
                    color='#2ecc71', alpha=0.7)
        axes[0].set_yticks(range(20))
        axes[0].set_yticklabels([f"Token {int(t)}" for t in top_20['token_id'].values])
        axes[0].set_xlabel('Transformer Ratio')
        axes[0].set_title('Top 20 Transformer-Preferring Tokens')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Plot top Mamba-preferring
        axes[1].barh(range(20), bottom_20['mean_transformer_ratio'].values,
                    xerr=bottom_20['std_transformer_ratio'].values,
                    color='#e74c3c', alpha=0.7)
        axes[1].set_yticks(range(20))
        axes[1].set_yticklabels([f"Token {int(t)}" for t in bottom_20['token_id'].values])
        axes[1].set_xlabel('Transformer Ratio')
        axes[1].set_title('Top 20 Mamba-Preferring Tokens')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'token_preferences.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze routing behavior of hybrid model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Config file')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./routing_analysis', help='Output directory')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to analyze')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = HybridTransformerSSM(
        vocab_size=config['model'].get('vocab_size', 1000),
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        num_classes=config['model']['num_classes'],
        dropout=config['model'].get('dropout', 0.1),
        max_seq_len=config['model'].get('max_seq_len', 16384),
        scheduler_temperature=config['scheduler'].get('temperature', 1.0),
        scheduler_hard=config['scheduler'].get('hard', False),
        scheduler_token_level=config['scheduler'].get('token_level', True),
        pos_encoding=config['model'].get('pos_encoding', 'sinusoidal')
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset_name=config['data']['dataset'],
        split='test',
        batch_size=config['training']['batch_size'],
        max_seq_len=config['model'].get('max_seq_len', 4096),
        num_workers=0,
        distributed=False,
        data_dir=args.data_dir
    )
    
    # Analyze routing
    analyzer = RoutingAnalyzer(model, device, args.output_dir)
    report = analyzer.generate_comprehensive_report(dataloader)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
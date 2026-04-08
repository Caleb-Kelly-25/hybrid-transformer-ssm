import itertools
import subprocess
import json
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class AblationStudy:
    """Run ablation studies for hybrid model."""
    
    def __init__(self, base_config_path: str = 'configs/base.yaml'):
        with open(base_config_path, 'r') as f:
            import yaml
            self.base_config = yaml.safe_load(f)
            
        self.results = []
        
    def run_scheduler_ablations(self):
        """Compare different scheduler configurations."""
        variants = [
            {'name': 'token_level_soft', 'scheduler_token_level': True, 'scheduler_hard': False},
            {'name': 'token_level_hard', 'scheduler_token_level': True, 'scheduler_hard': True},
            {'name': 'sequence_level_soft', 'scheduler_token_level': False, 'scheduler_hard': False},
            {'name': 'sequence_level_hard', 'scheduler_token_level': False, 'scheduler_hard': False},
        ]
        
        return self._run_variants('scheduler_ablation', variants)
    
    def run_architecture_ablations(self):
        """Compare different architectural choices."""
        variants = [
            {'name': 'baseline_transformer', 'use_ssm': False},
            {'name': 'baseline_mamba', 'use_transformer': False},
            {'name': 'static_hybrid_50_50', 'static_ratio': 0.5},
            {'name': 'adaptive_hybrid', 'use_scheduler': True},
        ]
        
        return self._run_variants('architecture_ablation', variants)
    
    def run_positional_encoding_ablations(self):
        """Compare different positional encodings."""
        variants = [
            {'name': 'sinusoidal', 'pos_encoding': 'sinusoidal'},
            {'name': 'rope', 'pos_encoding': 'rope'},
            {'name': 'learnable', 'pos_encoding': 'learnable'},
            {'name': 'none', 'pos_encoding': 'none'},
        ]
        
        return self._run_variants('pos_encoding_ablation', variants)
    
    def run_depth_ablations(self):
        """Compare different model depths."""
        variants = [
            {'name': 'depth_2', 'depth': 2},
            {'name': 'depth_4', 'depth': 4},
            {'name': 'depth_6', 'depth': 6},
            {'name': 'depth_8', 'depth': 8},
            {'name': 'depth_12', 'depth': 12},
        ]
        
        return self._run_variants('depth_ablation', variants)
    
    def run_sequence_length_analysis(self):
        """Analyze performance across different sequence lengths."""
        variants = [
            {'name': 'len_512', 'max_seq_len': 512},
            {'name': 'len_1024', 'max_seq_len': 1024},
            {'name': 'len_2048', 'max_seq_len': 2048},
            {'name': 'len_4096', 'max_seq_len': 4096},
            {'name': 'len_8192', 'max_seq_len': 8192},
            {'name': 'len_16384', 'max_seq_len': 16384},
        ]
        
        return self._run_variants('sequence_length_analysis', variants)
    
    def _run_variants(self, name: str, variants: List[Dict[str, Any]]):
        """Run experiments for each variant."""
        results = []
        
        for variant in variants:
            # Update config
            config = self.base_config.copy()
            for key, value in variant.items():
                if key.startswith('scheduler_'):
                    config['scheduler'][key.replace('scheduler_', '')] = value
                elif key in ['depth', 'pos_encoding', 'max_seq_len']:
                    config['model'][key] = value
                else:
                    config[key] = value
            
            # Save temp config
            config_path = f'temp_{name}_{variant["name"]}.yaml'
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(config, f)
            
            # Run experiment
            cmd = f"python experiments/run_experiment.py --config {config_path}"
            subprocess.run(cmd, shell=True)
            
            # Load results
            with open(f'logs/{name}_{variant["name"]}_results.json', 'r') as f:
                result = json.load(f)
                result['variant'] = variant['name']
                results.append(result)
                
        # Save combined results
        with open(f'logs/{name}_combined_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def create_ablation_plots(self):
        """Create visualization of ablation results."""
        # Load all results
        all_results = []
        import glob
        for result_file in glob.glob('logs/*_combined_results.json'):
            with open(result_file, 'r') as f:
                results = json.load(f)
                ablation_type = result_file.split('/')[-1].replace('_combined_results.json', '')
                for r in results:
                    r['ablation_type'] = ablation_type
                    all_results.append(r)
        
        df = pd.DataFrame(all_results)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        ablation_types = df['ablation_type'].unique()
        
        for idx, ablation in enumerate(ablation_types):
            ax = axes[idx // 2, idx % 2]
            ablation_df = df[df['ablation_type'] == ablation]
            
            # Bar plot of accuracy
            sns.barplot(data=ablation_df, x='variant', y='accuracy', ax=ax)
            ax.set_title(f'{ablation} Ablation Results')
            ax.set_xlabel('Variant')
            ax.set_ylabel('Accuracy')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(ablation_df['accuracy']):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('figures/ablation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary table
        summary = df.groupby(['ablation_type', 'variant']).agg({
            'accuracy': ['mean', 'std'],
            'inference_time': 'mean',
            'memory_usage': 'mean',
            'transformer_ratio': 'mean'
        }).round(4)
        
        print("\nAblation Study Summary:")
        print(summary)
        
        return summary


if __name__ == "__main__":
    study = AblationStudy()
    
    # Run all ablations
    print("Running scheduler ablations...")
    study.run_scheduler_ablations()
    
    print("Running architecture ablations...")
    study.run_architecture_ablations()
    
    print("Running positional encoding ablations...")
    study.run_positional_encoding_ablations()
    
    print("Running depth ablations...")
    study.run_depth_ablations()
    
    print("Running sequence length analysis...")
    study.run_sequence_length_analysis()
    
    # Create plots
    study.create_ablation_plots()
import argparse
import yaml
import torch
import wandb
from pathlib import Path
import json

from models.hybrid_model import HybridTransformerSSM
from data.dataloaders import create_dataloader
from training.trainer import HybridTrainer
from evaluation.visualization import RoutingVisualizer
from evaluation.benchmark import benchmark_model
from utils.logging_utils import setup_logging


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    # Setup logging
    logger = setup_logging(config.get('log_dir', './logs'))
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = create_dataloader(
        dataset_name=config['dataset'],
        split='train',
        batch_size=config['batch_size'],
        max_seq_len=config.get('max_seq_len', 4096),
        num_workers=config.get('num_workers', 4),
        distributed=False,
        data_dir=config.get('data_dir', './data'),
        **config.get('dataset_kwargs', {})
    )
    
    val_loader = create_dataloader(
        dataset_name=config['dataset'],
        split='val',
        batch_size=config['batch_size'],
        max_seq_len=config.get('max_seq_len', 4096),
        num_workers=config.get('num_workers', 4),
        distributed=False,
        data_dir=config.get('data_dir', './data'),
        **config.get('dataset_kwargs', {})
    )
    
    test_loader = create_dataloader(
        dataset_name=config['dataset'],
        split='test',
        batch_size=config['batch_size'],
        max_seq_len=config.get('max_seq_len', 4096),
        num_workers=config.get('num_workers', 4),
        distributed=False,
        data_dir=config.get('data_dir', './data'),
        **config.get('dataset_kwargs', {})
    )
    
    # Create model
    logger.info("Creating model...")
    model = HybridTransformerSSM(
        vocab_size=config.get('vocab_size', 1000),
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 16384),
        scheduler_temperature=config.get('scheduler_temperature', 1.0),
        scheduler_hard=config.get('scheduler_hard', False),
        scheduler_token_level=config.get('scheduler_token_level', True),
        pos_encoding=config.get('pos_encoding', 'sinusoidal'),
        checkpoint_activations=config.get('checkpoint_activations', False)
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params:,} parameters")
    
    # Create trainer
    trainer = HybridTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # Train model
    if not args.evaluate_only:
        logger.info("Starting training...")
        trainer.train()
    
    # Evaluate and visualize
    if args.evaluate_only or args.visualize:
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # Benchmark
        logger.info("Running benchmarks...")
        benchmark_results = benchmark_model(
            model,
            test_loader,
            device=trainer.device
        )
        
        logger.info(f"Benchmark Results:")
        for k, v in benchmark_results.items():
            logger.info(f"  {k}: {v}")
        
        # Visualize routing
        if args.visualize:
            logger.info("Creating visualizations...")
            visualizer = RoutingVisualizer(save_dir=config.get('figure_dir', './figures'))
            
            # Get a batch for visualization
            batch = next(iter(test_loader))
            batch = {k: v.to(trainer.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    lengths=batch['lengths'],
                    return_routing=True
                )
            
            # Create visualizations
            visualizer.plot_routing_heatmap(
                outputs['routing_weights'],
                batch['input_ids'],
                save_name=f"routing_heatmap_{config['dataset']}.png"
            )
            
            visualizer.plot_layer_routing_distribution(
                outputs['routing_weights'],
                save_name=f"layer_distribution_{config['dataset']}.png"
            )
            
            logger.info(f"Visualizations saved to {config.get('figure_dir', './figures')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid Transformer-SSM model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate model')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for evaluation')
    
    args = parser.parse_args()
    main(args)
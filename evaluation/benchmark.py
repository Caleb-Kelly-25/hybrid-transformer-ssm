"""
Comprehensive benchmarking suite for hybrid models.
Measures inference time, memory usage, throughput, and accuracy.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
from pathlib import Path
import psutil
import GPUtil


class ModelBenchmark:
    """Comprehensive model benchmarking."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100
    ):
        self.model = model
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        
        self.model.eval()
        
    def benchmark_inference(
        self,
        dataloader: torch.utils.data.DataLoader,
        sequence_lengths: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive inference benchmarking.
        
        Returns:
            Dictionary with timing, memory, and throughput metrics
        """
        print("Running inference benchmark...")
        
        metrics = {
            'inference_times': [],
            'memory_usage': [],
            'throughput': [],
            'latency_per_token': []
        }
        
        with torch.no_grad():
            # Warmup
            print(f"Warming up ({self.warmup_iterations} iterations)...")
            for i, batch in enumerate(dataloader):
                if i >= self.warmup_iterations:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                _ = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
            
            # Benchmark
            print(f"Benchmarking ({self.benchmark_iterations} iterations)...")
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            for i, batch in enumerate(tqdm(dataloader, total=self.benchmark_iterations)):
                if i >= self.benchmark_iterations:
                    break
                    
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].shape[0]
                seq_len = batch['input_ids'].shape[1]
                
                # Memory tracking
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated()
                
                # Timing
                start_time = time.time()
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Memory tracking
                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated()
                    peak_mem = torch.cuda.max_memory_allocated()
                    memory_used = peak_mem - mem_before
                    metrics['memory_usage'].append(memory_used / 1024**2)  # MB
                
                # Timing metrics
                inference_time = end_time - start_time
                metrics['inference_times'].append(inference_time)
                
                # Throughput
                tokens_processed = batch_size * seq_len
                throughput = tokens_processed / inference_time
                metrics['throughput'].append(throughput)
                
                # Latency per token
                latency_per_token = inference_time / tokens_processed * 1000  # ms
                metrics['latency_per_token'].append(latency_per_token)
        
        # Compute summary statistics
        summary = {}
        for key, values in metrics.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
                summary[f'{key}_p95'] = np.percentile(values, 95)
                summary[f'{key}_p99'] = np.percentile(values, 99)
        
        return summary
    
    def benchmark_memory_scaling(
        self,
        input_generator,
        sequence_lengths: List[int] = [512, 1024, 2048, 4096, 8192, 16384],
        batch_size: int = 1
    ) -> Dict[int, Dict[str, float]]:
        """
        Measure memory usage across different sequence lengths.
        """
        print("Running memory scaling benchmark...")
        
        memory_stats = {}
        
        with torch.no_grad():
            for seq_len in sequence_lengths:
                print(f"Testing sequence length: {seq_len}")
                
                # Generate input
                input_ids = input_generator(batch_size, seq_len).to(self.device)
                attention_mask = torch.ones_like(input_ids)
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                
                # Forward pass
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                    current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                else:
                    # CPU memory tracking
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    peak_memory = memory_info.peak_wset / 1024**2  # MB
                    current_memory = memory_info.rss / 1024**2  # MB
                
                memory_stats[seq_len] = {
                    'peak_memory_mb': peak_memory,
                    'current_memory_mb': current_memory,
                    'memory_per_token_bytes': (peak_memory * 1024**2) / (batch_size * seq_len)
                }
                
                # Try to detect if memory scales linearly
                if len(memory_stats) > 1:
                    prev_len = list(memory_stats.keys())[-2]
                    memory_ratio = peak_memory / memory_stats[prev_len]['peak_memory_mb']
                    length_ratio = seq_len / prev_len
                    memory_stats[seq_len]['scaling_factor'] = memory_ratio / length_ratio
        
        return memory_stats
    
    def benchmark_throughput_scaling(
        self,
        input_generator,
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        seq_len: int = 2048,
        iterations: int = 50
    ) -> Dict[int, Dict[str, float]]:
        """
        Measure throughput scaling with batch size.
        """
        print("Running throughput scaling benchmark...")
        
        throughput_stats = {}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Generate fixed input for fair comparison
            input_ids = input_generator(batch_size, seq_len).to(self.device)
            attention_mask = torch.ones_like(input_ids)
            
            # Warmup
            for _ in range(10):
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(iterations):
                    start = time.time()
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.time() - start)
            
            avg_time = np.mean(times)
            throughput = (batch_size * seq_len) / avg_time
            
            throughput_stats[batch_size] = {
                'avg_time_ms': avg_time * 1000,
                'throughput_tokens_per_sec': throughput,
                'throughput_samples_per_sec': batch_size / avg_time,
                'efficiency': throughput / batch_size  # Normalized throughput
            }
        
        return throughput_stats
    
    def benchmark_accuracy_vs_speed(
        self,
        dataloader: torch.utils.data.DataLoader,
        model_variants: List[Tuple[str, torch.nn.Module]]
    ) -> pd.DataFrame:
        """
        Compare accuracy vs inference speed across model variants.
        """
        import pandas as pd
        
        print("Running accuracy vs speed benchmark...")
        
        results = []
        
        for name, model_variant in model_variants:
            print(f"Testing variant: {name}")
            
            model_variant.to(self.device)
            model_variant.eval()
            
            # Measure accuracy
            correct = 0
            total = 0
            
            # Measure speed
            times = []
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    start = time.time()
                    outputs = model_variant(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.time() - start)
                    
                    predictions = outputs['logits'].argmax(dim=-1)
                    correct += (predictions == batch['labels']).sum().item()
                    total += batch['labels'].size(0)
            
            accuracy = correct / total
            avg_time = np.mean(times)
            
            results.append({
                'model': name,
                'accuracy': accuracy,
                'inference_time_ms': avg_time * 1000,
                'params_millions': sum(p.numel() for p in model_variant.parameters()) / 1e6
            })
        
        df = pd.DataFrame(results)
        df['accuracy_per_ms'] = df['accuracy'] / df['inference_time_ms']
        
        return df
    
    def profile_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        save_dir: str = './profiling'
    ) -> Dict:
        """
        Profile model using PyTorch profiler.
        """
        from torch.profiler import profile, record_function, ProfilerActivity
        
        print("Running PyTorch profiler...")
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        batch = next(iter(dataloader))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        with profile(activities=activities, record_shapes=True) as prof:
            with record_function("model_inference"):
                _ = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
        
        # Save profiling results
        prof.export_chrome_trace(f"{save_dir}/trace.json")
        
        # Get summary
        summary = prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total")
        
        with open(f"{save_dir}/profiling_summary.txt", 'w') as f:
            f.write(summary)
        
        print(f"Profiling results saved to {save_dir}/")
        
        return {'trace_file': f"{save_dir}/trace.json", 'summary': summary}
    
    def generate_benchmark_report(
        self,
        dataloader: torch.utils.data.DataLoader,
        input_generator,
        save_dir: str = './benchmarks'
    ) -> Dict:
        """
        Generate comprehensive benchmark report.
        """
        print("\n" + "="*50)
        print("GENERATING COMPREHENSIVE BENCHMARK REPORT")
        print("="*50)
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        report = {}
        
        # 1. Inference benchmark
        print("\n1. Inference Performance")
        print("-" * 40)
        inference_metrics = self.benchmark_inference(dataloader)
        report['inference'] = inference_metrics
        
        print(f"  Mean inference time: {inference_metrics['inference_times_mean']*1000:.2f} ms")
        print(f"  Mean throughput: {inference_metrics['throughput_mean']:.0f} tokens/sec")
        print(f"  Mean memory: {inference_metrics.get('memory_usage_mean', 0):.1f} MB")
        
        # 2. Memory scaling
        print("\n2. Memory Scaling")
        print("-" * 40)
        memory_scaling = self.benchmark_memory_scaling(input_generator)
        report['memory_scaling'] = memory_scaling
        
        for length, stats in memory_scaling.items():
            print(f"  Length {length}: {stats['peak_memory_mb']:.1f} MB")
        
        # 3. Throughput scaling
        print("\n3. Throughput Scaling")
        print("-" * 40)
        throughput_scaling = self.benchmark_throughput_scaling(input_generator)
        report['throughput_scaling'] = throughput_scaling
        
        for bs, stats in throughput_scaling.items():
            print(f"  Batch size {bs}: {stats['throughput_tokens_per_sec']:.0f} tokens/sec")
        
        # 4. Save report
        with open(f"{save_dir}/benchmark_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # 5. Create plots
        self._plot_benchmark_results(report, save_dir)
        
        print(f"\nBenchmark report saved to {save_dir}/")
        
        return report
    
    def _plot_benchmark_results(self, report: Dict, save_dir: str):
        """Create visualization of benchmark results."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Memory scaling
        if 'memory_scaling' in report:
            lengths = list(report['memory_scaling'].keys())
            memories = [report['memory_scaling'][l]['peak_memory_mb'] for l in lengths]
            
            axes[0, 0].plot(lengths, memories, 'o-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Sequence Length')
            axes[0, 0].set_ylabel('Peak Memory (MB)')
            axes[0, 0].set_title('Memory vs Sequence Length')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xscale('log')
        
        # Throughput scaling
        if 'throughput_scaling' in report:
            batch_sizes = list(report['throughput_scaling'].keys())
            throughputs = [report['throughput_scaling'][bs]['throughput_tokens_per_sec'] for bs in batch_sizes]
            
            axes[0, 1].plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=8)
            axes[0, 1].set_xlabel('Batch Size')
            axes[0, 1].set_ylabel('Throughput (tokens/sec)')
            axes[0, 1].set_title('Throughput vs Batch Size')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Inference time distribution
        if 'inference' in report and 'inference_times_mean' in report['inference']:
            axes[1, 0].bar(['Mean', 'P95', 'P99'], 
                          [report['inference']['inference_times_mean'] * 1000,
                           report['inference']['inference_times_p95'] * 1000,
                           report['inference']['inference_times_p99'] * 1000])
            axes[1, 0].set_ylabel('Time (ms)')
            axes[1, 0].set_title('Inference Time Distribution')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Memory breakdown
        if 'inference' in report and 'memory_usage_mean' in report['inference']:
            axes[1, 1].bar(['Mean', 'Peak'], 
                          [report['inference']['memory_usage_mean'],
                           report['inference']['memory_usage_max']])
            axes[1, 1].set_ylabel('Memory (MB)')
            axes[1, 1].set_title('Memory Usage')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/benchmark_plots.png", dpi=300, bbox_inches='tight')
        plt.close()


def benchmark_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str = './benchmarks'
) -> Dict:
    """
    Convenience function for quick benchmarking.
    """
    # Simple input generator for scaling tests
    def input_generator(batch_size, seq_len):
        return torch.randint(0, 1000, (batch_size, seq_len))
    
    benchmark = ModelBenchmark(model, device)
    return benchmark.generate_benchmark_report(dataloader, input_generator, save_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--output_dir', type=str, default='./benchmarks')
    
    args = parser.parse_args()
    
    # Load model and run benchmark
    # (Implementation similar to analyze_routing.py)
    print("Benchmarking model...")
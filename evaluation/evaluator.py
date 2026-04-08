"""
Comprehensive evaluation framework for hybrid models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score
)
from tqdm import tqdm
import json
from pathlib import Path
import pandas as pd


class HybridEvaluator:
    """Comprehensive evaluation for hybrid models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: str = './evaluation'
    ):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        compute_routing_stats: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        """
        all_logits = []
        all_labels = []
        all_predictions = []
        routing_weights_list = []
        
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                lengths=batch['lengths'],
                return_routing=compute_routing_stats
            )
            
            logits = outputs['logits']
            loss = criterion(logits, batch['labels'])
            total_loss += loss.item()
            
            all_logits.append(logits.cpu())
            all_labels.append(batch['labels'].cpu())
            all_predictions.append(logits.argmax(dim=-1).cpu())
            
            if compute_routing_stats:
                routing_weights_list.extend(outputs['routing_weights'])
        
        # Concatenate all results
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        
        # Compute metrics
        metrics = self._compute_classification_metrics(
            all_labels.numpy(),
            all_predictions.numpy(),
            all_logits.numpy()
        )
        
        metrics['loss'] = total_loss / len(dataloader)
        
        # Add routing statistics
        if compute_routing_stats and routing_weights_list:
            routing_metrics = self._compute_routing_metrics(routing_weights_list)
            metrics.update(routing_metrics)
        
        return metrics
    
    def _compute_classification_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        logits: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics."""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        
        # Per-class and averaged metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        metrics['precision_macro'] = np.mean(precision)
        metrics['recall_macro'] = np.mean(recall)
        metrics['f1_macro'] = np.mean(f1)
        
        # Weighted averages
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        metrics['precision_weighted'] = precision_w
        metrics['recall_weighted'] = recall_w
        metrics['f1_weighted'] = f1_w
        
        # Per-class metrics
        for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
            metrics[f'class_{i}_precision'] = p
            metrics[f'class_{i}_recall'] = r
            metrics[f'class_{i}_f1'] = f
            metrics[f'class_{i}_support'] = s
        
        # AUC metrics for binary/multi-class
        if logits.shape[1] == 2:
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
            try:
                metrics['auc_roc'] = roc_auc_score(labels, probs)
                metrics['auc_pr'] = average_precision_score(labels, probs)
            except:
                pass
        elif logits.shape[1] > 2:
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            try:
                metrics['auc_roc_ovo'] = roc_auc_score(labels, probs, multi_class='ovo', average='macro')
                metrics['auc_roc_ovr'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            except:
                pass
        
        return metrics
    
    def _compute_routing_metrics(
        self,
        routing_weights: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute routing-specific metrics."""
        
        metrics = {}
        
        # Average across layers
        stacked_weights = torch.stack([w.cpu() for w in routing_weights])
        
        # Overall transformer ratio
        metrics['transformer_ratio_mean'] = stacked_weights[..., 0].mean().item()
        metrics['transformer_ratio_std'] = stacked_weights[..., 0].std().item()
        
        # Per-layer ratios
        for i in range(stacked_weights.shape[0]):
            metrics[f'layer_{i}_transformer_ratio'] = stacked_weights[i, ..., 0].mean().item()
            metrics[f'layer_{i}_transformer_std'] = stacked_weights[i, ..., 0].std().item()
        
        # Routing entropy
        entropy = -(stacked_weights * (stacked_weights + 1e-8).log()).sum(dim=-1).mean().item()
        metrics['routing_entropy'] = entropy
        
        # Routing confidence
        confidence = stacked_weights.max(dim=-1)[0].mean().item()
        metrics['routing_confidence'] = confidence
        
        # Sparsity (percentage of confident decisions)
        sparsity = (stacked_weights.max(dim=-1)[0] > 0.9).float().mean().item()
        metrics['routing_sparsity'] = sparsity
        
        return metrics
    
    def evaluate_per_sequence_length(
        self,
        dataloader: torch.utils.data.DataLoader,
        length_buckets: List[Tuple[int, int]] = [
            (0, 512), (512, 1024), (1024, 2048), 
            (2048, 4096), (4096, 8192), (8196, float('inf'))
        ]
    ) -> pd.DataFrame:
        """
        Evaluate model performance broken down by sequence length.
        """
        bucket_results = {f"{low}-{high if high != float('inf') else 'max'}": [] 
                        for low, high in length_buckets}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating by length"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                lengths = batch['lengths'].cpu().numpy()
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    lengths=batch['lengths'],
                    return_routing=False
                )
                
                predictions = outputs['logits'].argmax(dim=-1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                for i, length in enumerate(lengths):
                    # Find bucket
                    for (low, high), bucket_list in bucket_results.items():
                        if low <= length < high:
                            bucket_list.append({
                                'length': length,
                                'correct': predictions[i] == labels[i],
                                'label': labels[i],
                                'prediction': predictions[i]
                            })
                            break
        
        # Compute statistics per bucket
        results = []
        for bucket_name, bucket_data in bucket_results.items():
            if bucket_data:
                correct = sum(d['correct'] for d in bucket_data)
                total = len(bucket_data)
                
                results.append({
                    'bucket': bucket_name,
                    'accuracy': correct / total,
                    'num_samples': total,
                    'avg_length': np.mean([d['length'] for d in bucket_data])
                })
        
        df = pd.DataFrame(results)
        df.to_csv(self.save_dir / 'performance_by_length.csv', index=False)
        
        return df
    
    def evaluate_calibration(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_bins: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate model calibration (reliability diagrams).
        """
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating calibration"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                all_logits.append(outputs['logits'].cpu())
                all_labels.append(batch['labels'].cpu())
        
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        probs = torch.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(confidences.numpy(), predictions.numpy(), labels.numpy(), num_bins)
        
        # Maximum Calibration Error (MCE)
        mce = self._compute_mce(confidences.numpy(), predictions.numpy(), labels.numpy(), num_bins)
        
        # Brier score
        brier = self._compute_brier_score(probs.numpy(), labels.numpy())
        
        metrics = {
            'ece': ece,
            'mce': mce,
            'brier_score': brier,
            'avg_confidence': confidences.mean().item(),
            'confidence_std': confidences.std().item()
        }
        
        return metrics
    
    def _compute_ece(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        for i in range(num_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_mce(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int
    ) -> float:
        """Compute Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        mce = 0.0
        
        for i in range(num_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if in_bin.sum() > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _compute_brier_score(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute Brier score."""
        num_classes = probs.shape[1]
        labels_one_hot = np.eye(num_classes)[labels]
        return np.mean(np.sum((probs - labels_one_hot) ** 2, axis=1))
    
    def generate_evaluation_report(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_name: str = 'unknown'
    ) -> Dict:
        """
        Generate comprehensive evaluation report.
        """
        print("\n" + "="*50)
        print(f"EVALUATION REPORT: {dataset_name}")
        print("="*50)
        
        report = {}
        
        # 1. Overall metrics
        print("\n1. Overall Performance")
        print("-" * 40)
        overall_metrics = self.evaluate(dataloader, compute_routing_stats=True)
        report['overall'] = overall_metrics
        
        print(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
        print(f"  F1 Score (macro): {overall_metrics['f1_macro']:.4f}")
        print(f"  Loss: {overall_metrics['loss']:.4f}")
        
        # 2. Routing statistics
        print("\n2. Routing Behavior")
        print("-" * 40)
        print(f"  Transformer ratio: {overall_metrics['transformer_ratio_mean']:.3f} ± {overall_metrics['transformer_ratio_std']:.3f}")
        print(f"  Routing entropy: {overall_metrics['routing_entropy']:.3f}")
        print(f"  Routing confidence: {overall_metrics['routing_confidence']:.3f}")
        
        # 3. Performance by length
        print("\n3. Performance by Sequence Length")
        print("-" * 40)
        length_performance = self.evaluate_per_sequence_length(dataloader)
        report['length_performance'] = length_performance.to_dict('records')
        
        for _, row in length_performance.iterrows():
            print(f"  {row['bucket']}: {row['accuracy']:.4f} (n={row['num_samples']})")
        
        # 4. Calibration
        print("\n4. Model Calibration")
        print("-" * 40)
        calibration = self.evaluate_calibration(dataloader)
        report['calibration'] = calibration
        
        print(f"  ECE: {calibration['ece']:.4f}")
        print(f"  MCE: {calibration['mce']:.4f}")
        print(f"  Brier Score: {calibration['brier_score']:.4f}")
        
        # Save report
        with open(self.save_dir / f'evaluation_report_{dataset_name}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to {self.save_dir}/")
        
        return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--output_dir', type=str, default='./evaluation')
    parser.add_argument('--dataset', type=str, default='test')
    
    args = parser.parse_args()
    
    # Load model and evaluate
    print("Evaluating model...")
import torch
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    routing_weights: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    """
    
    # Predictions
    predictions = logits.argmax(dim=-1)
    
    # Classification metrics
    accuracy = accuracy_score(labels.cpu(), predictions.cpu())
    
    # Multi-class metrics (macro average)
    if logits.size(-1) > 2:
        f1 = f1_score(labels.cpu(), predictions.cpu(), average='macro')
        precision = precision_score(labels.cpu(), predictions.cpu(), average='macro')
        recall = recall_score(labels.cpu(), predictions.cpu(), average='macro')
    else:
        f1 = f1_score(labels.cpu(), predictions.cpu())
        precision = precision_score(labels.cpu(), predictions.cpu())
        recall = recall_score(labels.cpu(), predictions.cpu())
    
    # Routing statistics
    transformer_ratios = []
    routing_entropies = []
    routing_confidences = []
    
    for weights in routing_weights:
        # Transformer usage ratio
        t_ratio = weights[..., 0].mean().item()
        transformer_ratios.append(t_ratio)
        
        # Routing entropy
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item()
        routing_entropies.append(entropy)
        
        # Routing confidence (max probability)
        confidence = weights.max(dim=-1)[0].mean().item()
        routing_confidences.append(confidence)
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'transformer_ratio': sum(transformer_ratios) / len(transformer_ratios),
        'routing_entropy': sum(routing_entropies) / len(routing_entropies),
        'routing_confidence': sum(routing_confidences) / len(routing_confidences)
    }
    
    return metrics
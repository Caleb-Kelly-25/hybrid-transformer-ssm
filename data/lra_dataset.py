import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, Tuple
import pickle
import os

class LRADataset(Dataset):
    """
    Long Range Arena dataset wrapper.
    Supports: ListOps, Text, Retrieval, Pathfinder, Path-X
    """
    
    TASKS = ['listops', 'text', 'retrieval', 'pathfinder', 'pathx']
    
    def __init__(
        self,
        task: str,
        split: str = 'train',
        max_seq_len: int = 16384,
        data_dir: str = './data/lra_release'
    ):
        assert task in self.TASKS, f"Task must be one of {self.TASKS}"
        
        self.task = task
        self.split = split
        self.max_seq_len = max_seq_len
        
        # Load data
        data_path = os.path.join(data_dir, f'{task}.{split}.pickle')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"LRA data not found at {data_path}. "
                                  f"Download from: https://github.com/google-research/long-range-arena")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.inputs = data['inputs']
        self.labels = data['labels']
        
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.inputs[idx][:self.max_seq_len], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': label,
            'attention_mask': attention_mask,
            'length': torch.tensor(input_ids.size(0))
        }


class SyntheticLongSequenceDataset(Dataset):
    """Synthetic dataset for long sequence modeling experiments."""
    
    def __init__(
        self,
        num_samples: int = 10000,
        seq_len: int = 4096,
        vocab_size: int = 1000,
        num_classes: int = 10,
        pattern_type: str = 'copy'  # 'copy', 'associative', 'hierarchy'
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.pattern_type = pattern_type
        
        # Generate data
        self.data = self._generate_data()
        
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        np.random.seed(42)
        
        if self.pattern_type == 'copy':
            # Copy task: model must copy a token from beginning to end
            inputs = np.random.randint(0, self.vocab_size, (self.num_samples, self.seq_len))
            targets = inputs[:, 0]  # Copy first token
            
        elif self.pattern_type == 'associative':
            # Associative recall: find matching token
            inputs = np.random.randint(0, self.vocab_size, (self.num_samples, self.seq_len))
            # Insert key-value pairs
            for i in range(self.num_samples):
                key = np.random.randint(0, self.vocab_size)
                value = np.random.randint(0, self.vocab_size)
                inputs[i, 0] = key
                inputs[i, 10] = value
                inputs[i, -10] = key
            targets = np.array([inputs[i, 10] for i in range(self.num_samples)])
            
        elif self.pattern_type == 'hierarchy':
            # Hierarchical structure task
            inputs = np.zeros((self.num_samples, self.seq_len), dtype=np.int32)
            for i in range(self.num_samples):
                depth = np.random.randint(1, 5)
                inputs[i, :depth] = 1
                inputs[i, -depth:] = 2
            targets = np.random.randint(0, self.num_classes, self.num_samples)
            
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")
            
        return inputs, targets
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.data[0][idx], dtype=torch.long),
            'labels': torch.tensor(self.data[1][idx], dtype=torch.long),
            'attention_mask': torch.ones(self.seq_len),
            'length': torch.tensor(self.seq_len)
        }
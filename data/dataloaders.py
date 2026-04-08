from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional
import torch

from .lra_dataset import LRADataset, SyntheticLongSequenceDataset


def create_dataloader(
    dataset_name: str,
    split: str = 'train',
    batch_size: int = 16,
    max_seq_len: int = 4096,
    num_workers: int = 4,
    distributed: bool = False,
    data_dir: str = './data',
    **kwargs
) -> DataLoader:
    """
    Create a dataloader for various datasets.
    """
    
    # Create dataset
    if dataset_name in LRADataset.TASKS:
        dataset = LRADataset(
            task=dataset_name,
            split=split,
            max_seq_len=max_seq_len,
            data_dir=data_dir
        )
    elif dataset_name == 'synthetic':
        dataset = SyntheticLongSequenceDataset(
            num_samples=kwargs.get('num_samples', 10000),
            seq_len=max_seq_len,
            vocab_size=kwargs.get('vocab_size', 1000),
            num_classes=kwargs.get('num_classes', 10),
            pattern_type=kwargs.get('pattern_type', 'copy')
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create sampler for distributed training
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=(split == 'train'))
        shuffle = False
    else:
        shuffle = (split == 'train')
    
    # Create dataloader with collate function for variable lengths
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_variable_length,
        drop_last=(split == 'train')
    )
    
    return dataloader


def collate_variable_length(batch):
    """Collate function for variable length sequences."""
    input_ids = [item['input_ids'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    lengths = torch.tensor([len(ids) for ids in input_ids])
    
    # Pad sequences
    max_len = max(lengths)
    padded_input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, (ids, length) in enumerate(zip(input_ids, lengths)):
        padded_input_ids[i, :length] = ids
        attention_mask[i, :length] = 1
        
    return {
        'input_ids': padded_input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'lengths': lengths
    }
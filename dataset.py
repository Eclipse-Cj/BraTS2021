import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

def pkload(file_path):
    """Load pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

class BraTSDataset(Dataset):
    """BraTS dataset for preprocessed data"""
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: Path to directory containing pickle files
            split: 'train' or 'test'
        """
        self.data_dir = data_dir
        
        # Read case names from text file
        split_file = os.path.join(data_dir, f'{split.upper()}', f'{split}.text')
        
        with open(split_file, 'r') as f:
            case_names = [line.strip() + '.pkl' for line in f]
        
        # Get full paths of pickle files
        self.file_paths = [os.path.join(data_dir, 'TRAIN' if split == 'train' else 'TEST', f) 
                           for f in case_names]
        
        print(f"Total {split} samples: {len(self.file_paths)}")
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load data
        data_path = self.file_paths[idx]
        
        try:
            data = pkload(data_path)
            
            # Expect data to be a dictionary from preprocessing
            image = data['image']
            label = data['label']
            
            return image, label
            
        except Exception as e:
            print(f"Error loading {data_path}: {str(e)}")
            # Return a dummy tensor in case of error
            return torch.zeros((4, 128, 128, 128)), torch.zeros((128, 128, 128))

def get_loaders(config):
    """Get data loaders for training and testing"""
    train_dataset = BraTSDataset(config.DATA_ROOT, split='train')
    test_dataset = BraTSDataset(config.DATA_ROOT, split='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Use batch size 1 for testing
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, test_loader
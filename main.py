import os
import argparse
import torch
import numpy as np
import random

from config import config
from model import VoluFormerUNet
from dataset import get_loaders
from train import train
from validate import validate_model, load_model_for_inference
from utils import load_checkpoint

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='VoluFormer-UNet for Brain Tumor Segmentation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate'],
                       help='Mode: train or validate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for validation')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to preprocessed data directory')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Update data directory if provided
    if args.data_dir:
        config.DATA_ROOT = args.data_dir
    
    # Print config information
    print(f"Using device: {config.DEVICE}")
    print(f"Data directory: {config.DATA_ROOT}")
    
    # Create model
    print("Creating model...")
    model = VoluFormerUNet(config)
    
    # Get data loaders
    print("Loading data...")
    train_loader, test_loader = get_loaders(config)
    
    if args.mode == 'train':
        # Train the model
        train(model, train_loader, test_loader, config, args.resume)
    
    elif args.mode == 'validate':
        # Set checkpoint path
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        # Load model for validation
        model = load_model_for_inference(model, checkpoint_path, config.DEVICE)
        
        # Run validation
        output_dir = os.path.join(config.RESULT_DIR, 'test')
        metrics = validate_model(model, test_loader, config.DEVICE, output_dir)

if __name__ == '__main__':
    main()
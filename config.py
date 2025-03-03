import os
import torch
from datetime import datetime

class Config:
    # Paths
    DATA_ROOT = 'E:\CJ\Brain_Tumor\pickledata'  # Change to your preprocessed data path
    OUTPUT_DIR = './output'
    
    # Create timestamp for the run
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Data parameters
    PATCH_SIZE = (128, 128, 128)
    IN_CHANNELS = 4  # T1, T1ce, T2, FLAIR
    OUT_CHANNELS = 4  # Background + 3 tumor regions
    
    
    # Model parameters
    DROPOUT_RATE = 0.2  # Dropout rate
    WINDOW_SIZE  = 4
    NUM_HEADS=[4, 4, 4, 4]
    EMDED_DIM = 32
    
    # Training parameters
    BATCH_SIZE = 1 
    NUM_WORKERS = 4  # Number of data loading workers
    NUM_EPOCHS = 300
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Loss weights
    DICE_WEIGHT = 1.0
    CE_WEIGHT = 1.0
    
    # Validation
    VAL_INTERVAL = 5  # Validate every N epochs
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create necessary directories
    def __init__(self):
        self.CHECKPOINT_DIR = os.path.join(self.OUTPUT_DIR, 'checkpoints')
        self.LOG_DIR = os.path.join(self.OUTPUT_DIR, 'logs')
        self.RESULT_DIR = os.path.join(self.OUTPUT_DIR, 'results')
        
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.RESULT_DIR, exist_ok=True)

config = Config()
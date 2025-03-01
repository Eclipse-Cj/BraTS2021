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
    NUM_CLASSES = 4  # Background + 3 tumor regions

    
    # Model parameters
    BASE_FILTERS = 8  # Base number of filters
    TRANSFORMER_DIM = 128  # Transformer embedding dimension
    TRANSFORMER_HEADS = 8  # Number of attention heads
    TRANSFORMER_LAYERS = 2  # Number of transformer layers
    DROPOUT_RATE = 0.1  # Dropout rate
    
    # Training parameters
    BATCH_SIZE = 1  # Adjust based on GPU memory
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
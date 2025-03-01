import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

def save_checkpoint(model, optimizer, epoch, val_score, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_score': val_score
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0, 0.0
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    val_score = checkpoint.get('val_score', 0.0)
    
    print(f"Loaded checkpoint from epoch {epoch} with validation score {val_score:.4f}")
    return epoch, val_score

def get_whole_tumor_mask(label):
    """Returns whole tumor mask (all labels > 0)"""
    return (label > 0).float()

def get_tumor_core_mask(label):
    """Returns tumor core mask (labels 1 and 3)"""
    return ((label == 1) | (label == 3)).float()

def get_enhancing_tumor_mask(label):
    """Returns enhancing tumor mask (label 3)"""
    return (label == 3).float()

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient"""
    # Flatten the tensors
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    # Calculate intersection and union
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)
    
    # Calculate Dice coefficient
    return (2. * intersection + smooth) / (union + smooth)

def sensitivity(y_true, y_pred, smooth=1e-6):
    """Calculate sensitivity (recall)"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    # True positives
    tp = torch.sum(y_true_f * y_pred_f)
    # False negatives
    fn = torch.sum(y_true_f * (1 - y_pred_f))
    
    return (tp + smooth) / (tp + fn + smooth)

def specificity(y_true, y_pred, smooth=1e-6):
    """Calculate specificity"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    # True negatives
    tn = torch.sum((1 - y_true_f) * (1 - y_pred_f))
    # False positives
    fp = torch.sum((1 - y_true_f) * y_pred_f)
    
    return (tn + smooth) / (tn + fp + smooth)

def calculate_metrics(y_true, y_pred):
    """Calculate all metrics for whole tumor, tumor core, and enhancing tumor"""
    # Convert to float for calculations
    if y_true.dtype != torch.float32:
        y_true = y_true.float()
    if y_pred.dtype != torch.float32:
        y_pred = y_pred.float()
    
    # Get masks for different tumor regions
    y_true_wt = get_whole_tumor_mask(y_true)
    y_pred_wt = get_whole_tumor_mask(y_pred)
    
    y_true_tc = get_tumor_core_mask(y_true)
    y_pred_tc = get_tumor_core_mask(y_pred)
    
    y_true_et = get_enhancing_tumor_mask(y_true)
    y_pred_et = get_enhancing_tumor_mask(y_pred)
    
    # Calculate Dice scores
    dice_wt = dice_coefficient(y_true_wt, y_pred_wt).item()
    dice_tc = dice_coefficient(y_true_tc, y_pred_tc).item()
    dice_et = dice_coefficient(y_true_et, y_pred_et).item()
    dice_mean = (dice_wt + dice_tc + dice_et) / 3.0
    
    # Calculate sensitivity
    sens_wt = sensitivity(y_true_wt, y_pred_wt).item()
    sens_tc = sensitivity(y_true_tc, y_pred_tc).item()
    sens_et = sensitivity(y_true_et, y_pred_et).item()
    sens_mean = (sens_wt + sens_tc + sens_et) / 3.0
    
    # Calculate specificity
    spec_wt = specificity(y_true_wt, y_pred_wt).item()
    spec_tc = specificity(y_true_tc, y_pred_tc).item()
    spec_et = specificity(y_true_et, y_pred_et).item()
    spec_mean = (spec_wt + spec_tc + spec_et) / 3.0
    
    return {
        'dice': {
            'wt': dice_wt,
            'tc': dice_tc,
            'et': dice_et,
            'mean': dice_mean
        },
        'sensitivity': {
            'wt': sens_wt,
            'tc': sens_tc,
            'et': sens_et,
            'mean': sens_mean
        },
        'specificity': {
            'wt': spec_wt,
            'tc': spec_tc,
            'et': spec_et,
            'mean': spec_mean
        }
    }

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # SAFETY CHECK: Ensure target values are within valid range
        # Clamp targets to be within [0, num_classes-1]
        targets_clamped = torch.clamp(targets, min=0, max=num_classes-1)
        
        # One-hot encode the targets
        targets_one_hot = torch.nn.functional.one_hot(targets_clamped, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Compute the Dice score for each class
        dice_scores = []
        for c in range(num_classes):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum(dim=[1, 2, 3])
            union = pred_c.sum(dim=[1, 2, 3]) + target_c.sum(dim=[1, 2, 3])
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice.mean())
        
        # Apply class weights if provided
        if self.weight is not None:
            weighted_dice = torch.stack(dice_scores) * self.weight
            return 1.0 - weighted_dice.sum() / self.weight.sum()
        else:
            # Skip background class (index 0) and focus on tumor classes
            tumor_dice = torch.stack(dice_scores[1:]).mean()
            return 1.0 - tumor_dice
        
def visualize_results(image, target, prediction, slice_idx=None, save_path=None):
    """
    Visualize segmentation results without using medical image background
    
    Key Features:
    - Handles different tensor shapes (with/without batch dimension)
    - Selects a mid-slice for visualization
    - Uses pure color mapping for segmentation areas
    """
    
    # Dynamically select mid-slice based on tensor dimensions
    if slice_idx is None:
        slice_idx = image.shape[2] // 2 if len(image.shape) == 4 else image.shape[3] // 2
    
    # Extract target and prediction slices with flexible tensor handling
    # Supports different tensor shapes and batch dimensions
    if len(target.shape) > 3:
        target_slice = target[0, slice_idx, :, :].cpu().numpy() if target.dim() == 4 else target[0, 0, slice_idx, :, :].cpu().numpy()
    else:
        target_slice = target[slice_idx, :, :].cpu().numpy()
    
    if len(prediction.shape) > 3:
        prediction_slice = prediction[0, slice_idx, :, :].cpu().numpy() if prediction.dim() == 4 else prediction[0, 0, slice_idx, :, :].cpu().numpy()
    else:
        prediction_slice = prediction[slice_idx, :, :].cpu().numpy()
    
    # Visualization setup
    plt.figure(figsize=(12, 5))
    
    # Ground Truth Visualization
    plt.subplot(121)
    # Color mapping for segmentation areas
    # 1 (Red): Necrotic core
    # 2 (Green): Edema
    # 3 (Blue): Enhancing tumor
    target_mask = np.zeros((*target_slice.shape, 3))
    target_mask[target_slice == 1, 0] = 1.0  # Red for necrotic core
    target_mask[target_slice == 2, 1] = 1.0  # Green for edema
    target_mask[target_slice == 3, 2] = 1.0  # Blue for enhancing tumor
    
    plt.imshow(target_mask)
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Prediction Visualization (identical approach)
    plt.subplot(122)
    pred_mask = np.zeros((*prediction_slice.shape, 3))
    pred_mask[prediction_slice == 1, 0] = 1.0  # Red for necrotic core
    pred_mask[prediction_slice == 2, 1] = 1.0  # Green for edema
    pred_mask[prediction_slice == 3, 2] = 1.0  # Blue for enhancing tumor
    
    plt.imshow(pred_mask)
    plt.title('Prediction')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save or display the figure
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
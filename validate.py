import os
import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
import pandas as pd

from config import config
from utils import AverageMeter, calculate_metrics, visualize_results

def validate_model(model, val_loader, device, output_dir):
    """Validate the model and save results"""
    model.eval()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    # Metrics
    all_metrics = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(val_loader, desc="Validating")):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions and calculate metrics
            preds = torch.argmax(outputs, dim=1)
            metrics = calculate_metrics(targets, preds)
            all_metrics.append(metrics)
            
            # Visualize results
            vis_path = os.path.join(vis_dir, f"sample_{i}.png")
            visualize_results(images, targets, preds, save_path=vis_path)
            
            # Save predictions as NIfTI
            pred_np = preds[0].cpu().numpy()
            
            # Convert back to original BraTS labels (3 -> 4)
            pred_np_brats = pred_np.copy()
            pred_np_brats[pred_np == 3] = 4
            
            # Create NIfTI file
            pred_nii = nib.Nifti1Image(pred_np_brats, np.eye(4))
            nib.save(pred_nii, os.path.join(pred_dir, f"pred_{i}.nii.gz"))
    
    # Calculate average metrics
    avg_metrics = {
        'dice': {
            'wt': np.mean([m['dice']['wt'] for m in all_metrics]),
            'tc': np.mean([m['dice']['tc'] for m in all_metrics]),
            'et': np.mean([m['dice']['et'] for m in all_metrics]),
            'mean': np.mean([m['dice']['mean'] for m in all_metrics])
        },
        'sensitivity': {
            'wt': np.mean([m['sensitivity']['wt'] for m in all_metrics]),
            'tc': np.mean([m['sensitivity']['tc'] for m in all_metrics]),
            'et': np.mean([m['sensitivity']['et'] for m in all_metrics]),
            'mean': np.mean([m['sensitivity']['mean'] for m in all_metrics])
        },
        'specificity': {
            'wt': np.mean([m['specificity']['wt'] for m in all_metrics]),
            'tc': np.mean([m['specificity']['tc'] for m in all_metrics]),
            'et': np.mean([m['specificity']['et'] for m in all_metrics]),
            'mean': np.mean([m['specificity']['mean'] for m in all_metrics])
        }
    }
    
    # Create DataFrame for metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Dice', 'Sensitivity', 'Specificity'],
        'Whole Tumor': [
            avg_metrics['dice']['wt'],
            avg_metrics['sensitivity']['wt'],
            avg_metrics['specificity']['wt']
        ],
        'Tumor Core': [
            avg_metrics['dice']['tc'],
            avg_metrics['sensitivity']['tc'],
            avg_metrics['specificity']['tc']
        ],
        'Enhancing Tumor': [
            avg_metrics['dice']['et'],
            avg_metrics['sensitivity']['et'],
            avg_metrics['specificity']['et']
        ],
        'Mean': [
            avg_metrics['dice']['mean'],
            avg_metrics['sensitivity']['mean'],
            avg_metrics['specificity']['mean']
        ]
    })
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Print results
    print("Validation Results:")
    print(metrics_df)
    
    return avg_metrics

def load_model_for_inference(model, checkpoint_path, device):
    """Load model weights for inference"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

def fixed_blocks_inference(model, x, device):
    """
    Process a large volume using 8 fixed overlapping blocks with specific stitching strategy.
    
    Args:
        model: The neural network model
        x: Input tensor [B, C, H, W, D]
        device: Device to run inference on
        
    Returns:
        Full volume prediction
    """
    # Get input shape
    B, C, H, W, D = x.shape
    
    # Create a tensor to hold the final output
    num_classes = 4  # Adjust based on your model's output classes
    y = torch.zeros((B, num_classes, H, W, D), device=device)
    
    # Make sure we're only processing one batch at a time
    assert B == 1, "This function only supports batch size 1"
    
    # Create the 8 fixed blocks as in TransBTS
    blocks = []
    # Front blocks (z: 0-128)
    blocks.append(x[..., :128, :128, :128])         # Left Upper Front
    blocks.append(x[..., :128, 112:240, :128])      # Right Upper Front
    blocks.append(x[..., 112:240, :128, :128])      # Left Lower Front
    blocks.append(x[..., 112:240, 112:240, :128])   # Right Lower Front
    
    # Back blocks (z: 27-155)
    blocks.append(x[..., :128, :128, 32:])          # Left Upper Back
    blocks.append(x[..., :128, 112:240, 32:])       # Right Upper Back
    blocks.append(x[..., 112:240, :128, 32:])       # Left Lower Back
    blocks.append(x[..., 112:240, 112:240, 32:])    # Right Lower Back
    
    # Process each block
    predictions = []
    for block in blocks:
        # Handle potential size issues - ensure block is exactly 128×128×128
        # If it's smaller in any dimension, pad it
        curr_h, curr_w, curr_d = block.shape[2:]
        if curr_h < 128 or curr_w < 128 or curr_d < 128:
            padded_block = torch.zeros((B, C, 128, 128, 128), device=device)
            padded_block[..., :curr_h, :curr_w, :curr_d] = block
            block = padded_block
        
        # Forward pass
        with torch.no_grad():
            pred = model(block)
        predictions.append(pred)
    
    # Stitch the predictions back together as in TransBTS
    # Front blocks
    y[..., :128, :128, :128] = predictions[0]
    y[..., :128, 128:240, :128] = predictions[1][..., :, 16:, :]
    y[..., 128:240, :128, :128] = predictions[2][..., 16:, :, :]
    y[..., 128:240, 128:240, :128] = predictions[3][..., 16:, 16:, :]
    
    # Back blocks
    z_end = min(D, 155)  # Make sure we don't exceed the actual depth
    y[..., :128, :128, 32:z_end] = predictions[4][..., :, :, :z_end-32]
    y[..., :128, 128:240, 32:z_end] = predictions[5][..., :, 16:, :z_end-32]
    y[..., 128:240, :128, 32:z_end] = predictions[6][..., 16:, :, :z_end-32]
    y[..., 128:240, 128:240, 32:z_end] = predictions[7][..., 16:, 16:, :z_end-32]
    
    # Crop to remove any additional padding in z direction
    z_size = min(D, 155)
    y = y[..., :z_size]
    
    return y

def validate_with_fixed_blocks(model, test_loader, criterion, epoch, device, writer, save_dir=None):
    """Validate the model using the 8-block fixed position approach"""
    model.eval()
    
    # Metrics
    loss_meter = AverageMeter()
    dice_wt_meter = AverageMeter()
    dice_tc_meter = AverageMeter()
    dice_et_meter = AverageMeter()
    dice_mean_meter = AverageMeter()
    
    # Use tqdm progress bar
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for i, (images, targets) in pbar:
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Process using 8-block approach
            outputs = fixed_blocks_inference(model, images, device)
            
            # Calculate loss (handle potential size mismatch)
            if outputs.shape[2:] != targets.shape[1:]:
                # Resize targets to match outputs
                targets_resized = F.interpolate(
                    targets.unsqueeze(1).float(), 
                    size=outputs.shape[2:], 
                    mode='nearest'
                ).squeeze(1).long()
                loss = criterion(outputs, targets_resized)
                
                # Use resized targets for metrics
                preds = torch.argmax(outputs, dim=1)
                metrics = calculate_metrics(targets_resized, preds)
            else:
                loss = criterion(outputs, targets)
                
                # Get predictions and calculate metrics
                preds = torch.argmax(outputs, dim=1)
                metrics = calculate_metrics(targets, preds)
            
            # Update meters
            loss_meter.update(loss.item())
            dice_wt_meter.update(metrics['dice']['wt'])
            dice_tc_meter.update(metrics['dice']['tc'])
            dice_et_meter.update(metrics['dice']['et'])
            dice_mean_meter.update(metrics['dice']['mean'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'dice_mean': f"{dice_mean_meter.avg:.4f}"
            })
            
            # Save visualization for the first few samples
            if i < 5 and save_dir:
                vis_path = os.path.join(save_dir, f"val_ep{epoch}_sample{i}.png")
                
                # Use the appropriate targets for visualization
                if outputs.shape[2:] != targets.shape[1:]:
                    visualize_results(images, targets_resized, preds, save_path=vis_path)
                else:
                    visualize_results(images, targets, preds, save_path=vis_path)
    
    # Log validation metrics
    writer.add_scalar('Loss/val', loss_meter.avg, epoch)
    writer.add_scalar('Dice/val/wt', dice_wt_meter.avg, epoch)
    writer.add_scalar('Dice/val/tc', dice_tc_meter.avg, epoch)
    writer.add_scalar('Dice/val/et', dice_et_meter.avg, epoch)
    writer.add_scalar('Dice/val/mean', dice_mean_meter.avg, epoch)
    
    # Print validation results
    print(f"\nValidation Epoch {epoch} - Loss: {loss_meter.avg:.4f}")
    print(f"Dice Scores - WT: {dice_wt_meter.avg:.4f}, TC: {dice_tc_meter.avg:.4f}, "
          f"ET: {dice_et_meter.avg:.4f}, Mean: {dice_mean_meter.avg:.4f}")
    
    # Return average metrics
    return {
        'loss': loss_meter.avg,
        'dice': {
            'wt': dice_wt_meter.avg,
            'tc': dice_tc_meter.avg,
            'et': dice_et_meter.avg,
            'mean': dice_mean_meter.avg
        }
    }
if __name__ == "__main__":
    # This allows the module to be imported without running
    pass
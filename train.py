import os
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from validate import fixed_blocks_inference
from config import config
from utils import (
    AverageMeter, calculate_metrics, save_checkpoint, 
    load_checkpoint, visualize_results, DiceLoss
)

def train_epoch(model, train_loader, criterion, optimizer, epoch, device, writer):
    """Train for one epoch"""
    model.train()
    
    # Metrics
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    
    # Use tqdm progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")
    
    for i, (images, targets) in pbar:
        # Move data to device
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Get predictions and calculate metrics
        preds = torch.argmax(outputs, dim=1)
        metrics = calculate_metrics(targets, preds)
        
        # Update meters
        loss_meter.update(loss.item())
        dice_meter.update(metrics['dice']['mean'])
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_meter.avg:.4f}",
            'dice': f"{dice_meter.avg:.4f}",
            'time': f"{batch_time.avg:.2f}s"
        })
        
        # Log to TensorBoard (every 20 steps)
        step = epoch * len(train_loader) + i
        if i % 20 == 0:
            writer.add_scalar('Loss/train_step', loss.item(), step)
            writer.add_scalar('Dice/train_step', metrics['dice']['mean'], step)
    
    # Log epoch metrics
    writer.add_scalar('Loss/train', loss_meter.avg, epoch)
    writer.add_scalar('Dice/train', dice_meter.avg, epoch)
    
    # Return average metrics
    return {
        'loss': loss_meter.avg,
        'dice': dice_meter.avg
    }

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

def train(model, train_loader, test_loader, config, resume_path=None):
    """Main training function"""
    # Setup device
    device = config.DEVICE
    model = model.to(device)
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    # Loss function and optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_dice = 0.0
    if resume_path:
        start_epoch, best_dice = load_checkpoint(model, optimizer, resume_path)
        start_epoch += 1
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # Train for one epoch
        train_metrics = train_epoch(
                    model, train_loader, criterion, optimizer, epoch, device, writer
                )
        
        # Validate every VAL_INTERVAL epochs or at the last epoch
        if (epoch + 1) % config.VAL_INTERVAL == 0 or epoch == config.NUM_EPOCHS - 1:
            val_metrics = validate_with_fixed_blocks(
                model, test_loader, criterion, epoch, device, writer, config.RESULT_DIR
            )
            
            # Update learning rate
            scheduler.step(val_metrics['dice']['mean'])
            
            # Save best model
            if val_metrics['dice']['mean'] > best_dice:
                best_dice = val_metrics['dice']['mean']
                save_checkpoint(
                    model, optimizer, epoch, best_dice,
                    os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
                )
                print(f"New best model saved with dice score: {best_dice:.4f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics['dice']['mean'],
                os.path.join(config.CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, config.NUM_EPOCHS-1, val_metrics['dice']['mean'],
        os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
    )
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    # This allows the module to be imported without running
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from swin import *

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m



class InitConv(nn.Module):
    """
    Input:
    (B, C, H, W, D): B-Batch size, C-Chaneels(4), H-height, W-weight, D-deepth
    """
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y
    
class DeBlock(nn.Module):
    """Decoder residual block similar to EnBlock"""
    def __init__(self, in_channels, norm='gn'):
        super(DeBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y

class DeUp(nn.Module):
    """Upsampling block for decoder"""
    def __init__(self, in_channels, out_channels):
        super(DeUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        y = self.conv(x)
        return y
    
class SwinUTransBTS(nn.Module):
    """
    Hybrid CNN-Transformer model for BraTS segmentation
    
    Architecture:
    - initial conv
    - cnn encoder
    - SwinTransformerBlock (no shift)
    - SwinTransformerBlock (shift)
    - cnn encoder
    - SwinTransformerBlock (no shift)
    - SwinTransformerBlock (shift)
    - cnn decoder
    - SwinTransformerBlock_kv (no shift)
    - SwinTransformerBlock_kv (shift)
    - cnn decoder
    - SwinTransformerBlock_kv (no shift)
    - SwinTransformerBlock_kv (shift)
    - Final layer (from TransBTS)
    """
    def __init__(self, in_channels=4, out_channels=4, embed_dim=48, window_size = 8, 
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), dropout=0.2):
        super(SwinUTransBTS, self).__init__()
        
        self.window_size = window_size
        self.embed_dim = embed_dim
        
        # Initial Convolution
        self.init_conv = InitConv(in_channels=in_channels, out_channels=embed_dim, dropout=dropout)
        
        # CNN Encoder 1
        self.encoder_block1 = EnBlock(embed_dim)
        self.down1 = EnDown(embed_dim, embed_dim*2)
        
        # Swin Transformer Block 1 (no shift)
        self.swin1 = SwinTransformerBlock(
            dim=embed_dim*2, 
            input_resolution=(64, 64, 64),  # Assuming 128x128x128 input, after down1
            num_heads=num_heads[0],
            window_size=window_size,
            shift_size=0
        )
        
        # Swin Transformer Block 2 (shift)
        self.swin2 = SwinTransformerBlock(
            dim=embed_dim*2, 
            input_resolution=(64, 64, 64),
            num_heads=num_heads[0],
            window_size=window_size,
            shift_size=window_size // 2
        )
        
        # CNN Encoder 2
        self.encoder_block2 = EnBlock(embed_dim*2)
        self.down2 = EnDown(embed_dim*2, embed_dim*4)
        
        # Swin Transformer Block 3 (no shift)
        self.swin3 = SwinTransformerBlock(
            dim=embed_dim*4, 
            input_resolution=(32, 32, 32),  # After down2
            num_heads=num_heads[1],
            window_size=window_size,
            shift_size=0
        )
        
        # Swin Transformer Block 4 (shift)
        self.swin4 = SwinTransformerBlock(
            dim=embed_dim*4, 
            input_resolution=(32, 32, 32),
            num_heads=num_heads[1],
            window_size=window_size,
            shift_size=window_size // 2
        )
        
        # Bottleneck
        self.bottleneck = EnBlock(embed_dim*4)
        
        # CNN Decoder 1
        self.up1 = DeUp(embed_dim*4, embed_dim*2)
        self.decoder_block1 = DeBlock(embed_dim*2)
        
        # Swin Transformer Block KV 1 (no shift)
        self.swin_kv1 = SwinTransformerBlock_kv(
            dim=embed_dim*2, 
            input_resolution=(64, 64, 64),
            num_heads=num_heads[0],
            window_size=window_size,
            shift_size=0
        )
        
        # Swin Transformer Block KV 2 (shift)
        self.swin_kv2 = SwinTransformerBlock_kv(
            dim=embed_dim*2, 
            input_resolution=(64, 64, 64),
            num_heads=num_heads[0],
            window_size=window_size,
            shift_size=window_size // 2
        )
        
        # CNN Decoder 2
        self.up2 = DeUp(embed_dim*2, embed_dim)
        self.decoder_block2 = DeBlock(embed_dim)
        
        # Swin Transformer Block KV 3 (no shift)
        self.swin_kv3 = SwinTransformerBlock_kv(
            dim=embed_dim, 
            input_resolution=(128, 128, 128),
            num_heads=max(1, num_heads[0]//3),  # Ensure at least 1 head
            window_size=window_size,
            shift_size=0
        )
        
        # Swin Transformer Block KV 4 (shift)
        self.swin_kv4 = SwinTransformerBlock_kv(
            dim=embed_dim, 
            input_resolution=(128, 128, 128),
            num_heads=max(1, num_heads[0]//3),
            window_size=window_size,
            shift_size=window_size // 2
        )
        
        # Final layer (from TransBTS)
        self.final_conv = nn.Sequential(
            # Convolution layer: convert feature channels to number of classes
            nn.Conv3d(embed_dim, out_channels, kernel_size=1),
            # Softmax: convert output to probability distribution
            nn.Softmax(dim=1)
        )
        
    def create_mask(self, x, H, W, D):
        """Create attention mask for shifted window attention"""
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        Dp = int(np.ceil(D / self.window_size)) * self.window_size
        
        img_mask = torch.zeros((1, Hp, Wp, Dp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.window_size//2),
                   slice(-self.window_size//2, None))
        w_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.window_size//2),
                   slice(-self.window_size//2, None))
        d_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.window_size//2),
                    slice(-self.window_size//2, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for d in d_slices:
                    img_mask[:, h, w, d, :] = cnt
                    cnt += 1
                    
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
        
    def forward(self, x):
        """Forward pass through the network"""
        # Input shape: [B, C, H, W, D]
        B, C, H, W, D = x.shape
        
        # Initial Convolution
        x = self.init_conv(x)  # B, embed_dim, 128, 128, 128
        
        # Encoder Stage 1
        skip1 = self.encoder_block1(x)  # B, embed_dim, 128, 128, 128
        x = self.down1(skip1)  # B, embed_dim*2, 64, 64, 64
        
        # Store dimensions for Transformer blocks
        B, C, Hs, Ws, Ds = x.shape
        
        # Prepare for Swin Transformer
        x_3d = x  # Store 5D tensor for later
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # B, Hs, Ws, Ds, C
        x = x.view(B, Hs * Ws * Ds, C)  # B, Hs*Ws*Ds, C
        
        # Create attention mask
        attn_mask_stage1 = self.create_mask(x_3d, Hs, Ws, Ds)
        
        # Swin Transformer Block 1 & 2
        x = self.swin1(x, attn_mask_stage1)
        x = self.swin2(x, attn_mask_stage1)
        
        # Reshape back to 5D
        x = x.view(B, Hs, Ws, Ds, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # B, C, Hs, Ws, Ds
        
        # Encoder Stage 2
        skip2 = x  # Save for skip connection
        x = self.encoder_block2(x)  # B, embed_dim*2, 64, 64, 64
        x = self.down2(x)  # B, embed_dim*4, 32, 32, 32
        
        # Store dimensions for Transformer blocks
        B, C, Hd, Wd, Dd = x.shape
        
        # Prepare for Swin Transformer
        x_3d = x  # Store 5D tensor for later
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # B, Hd, Wd, Dd, C
        x = x.view(B, Hd * Wd * Dd, C)  # B, Hd*Wd*Dd, C
        
        # Create attention mask
        attn_mask_stage2 = self.create_mask(x_3d, Hd, Wd, Dd)
        
        # Swin Transformer Block 3 & 4
        x = self.swin3(x, attn_mask_stage2)
        x = self.swin4(x, attn_mask_stage2)
        
        # Reshape back to 5D
        x = x.view(B, Hd, Wd, Dd, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # B, C, Hd, Wd, Dd
        
        # Bottleneck
        x = self.bottleneck(x)  # B, embed_dim*4, 32, 32, 32
        
        # Decoder Stage 1
        x = self.up1(x)  # B, embed_dim*2, 64, 64, 64
        x = x + skip2  # Skip connection, B, embed_dim*2, 64, 64, 64
        x = self.decoder_block1(x)  # B, embed_dim*2, 64, 64, 64
        
        # Prepare for Swin Transformer KV
        B, C, H, W, D = x.shape
        
        # Convert skip2 and x to required format for SwinTransformerBlock_kv
        x_3d = x
        skip2_3d = skip2
        
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # B, H, W, D, C
        x = x.view(B, H * W * D, C)  # B, H*W*D, C
        
        skip2 = skip2.permute(0, 2, 3, 4, 1).contiguous()  # B, H, W, D, C
        skip2 = skip2.view(B, H * W * D, C)  # B, H*W*D, C
        
        # Create attention mask
        attn_mask_stage1 = self.create_mask(x_3d, H, W, D)
        
        # Swin Transformer Block KV 1 & 2
        x = self.swin_kv1(x, attn_mask_stage1, skip2, x)
        x = self.swin_kv2(x, attn_mask_stage1, skip2, x)
        
        # Reshape back to 5D
        x = x.view(B, H, W, D, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # B, C, H, W, D
        
        # Decoder Stage 2
        x = self.up2(x)  # B, embed_dim, 128, 128, 128
        x = x + skip1  # Skip connection
        x = self.decoder_block2(x)  # B, embed_dim, 128, 128, 128
        
        # Prepare for Swin Transformer KV
        B, C, H, W, D = x.shape
        
        # Convert skip1 and x to required format for SwinTransformerBlock_kv
        x_3d = x
        skip1_3d = skip1
        
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # B, H, W, D, C
        x = x.view(B, H * W * D, C)  # B, H*W*D, C
        
        skip1 = skip1.permute(0, 2, 3, 4, 1).contiguous()  # B, H, W, D, C
        skip1 = skip1.view(B, H * W * D, C)  # B, H*W*D, C
        
        # Create attention mask for original size
        attn_mask_orig = self.create_mask(x_3d, H, W, D)
        
        # Swin Transformer Block KV 3 & 4
        x = self.swin_kv3(x, attn_mask_orig, skip1, x)
        x = self.swin_kv4(x, attn_mask_orig, skip1, x)
        
        # Reshape back to 5D
        x = x.view(B, H, W, D, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # B, C, H, W, D
        
        # Final layer (outputs segmentation with channels corresponding to classes)
        x = self.final_conv(x)  # B, out_channels, 128, 128, 128
        
        return x

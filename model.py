import torch
import torch.nn as nn
import torch.nn.functional as F

class InitialConvLayer(nn.Module):
    def __init__(self, in_channels=4, out_channels=16):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        # Input: [B, 4, 128, 128, 128] (B=batch_size)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        # Output: [B, 16, 128, 128, 128]
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First convolutional path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.activation1 = nn.LeakyReLU(0.2, inplace=True)
        
        # Second convolutional path
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.activation2 = nn.LeakyReLU(0.2, inplace=True)
        
        # Skip connection (identity or projection)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels)
            )
    
    def forward(self, x):
        # Input: [B, C_in, H, W, D]
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Skip connection
        identity = self.skip(identity)
        
        # Residual addition
        out += identity
        out = self.activation2(out)
        # Output: [B, C_out, H/stride, W/stride, D/stride]
        return out
    
class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2, downsample=True):
        super().__init__()
        layers = []
        
        # First block may downsample
        stride = 2 if downsample else 1
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        
        # Additional blocks at same resolution
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        self.blocks = nn.Sequential(*layers)
        self.dropout = nn.Dropout3d(0.2)
    
    def forward(self, x):
        # Input: [B, C_in, H, W, D]
        x = self.blocks(x)
        x = self.dropout(x)
        # Output: [B, C_out, H/2, W/2, D/2] if downsample=True
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=256, embed_dim=512, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv3d(in_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # Input: [B, C, H, W, D]
        B, C, H, W, D = x.shape
        # Project and reshape
        x = self.projection(x)  # [B, embed_dim, H/patch_size, W/patch_size, D/patch_size]
        x = x.flatten(2)        # [B, embed_dim, (H/patch_size)*(W/patch_size)*(D/patch_size)]
        x = x.permute(0, 2, 1)  # [B, (H/patch_size)*(W/patch_size)*(D/patch_size), embed_dim]
        return x
    
class PositionalEncoding3D(nn.Module):
    def __init__(self, embed_dim=512, max_len=64):
        super().__init__()
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        # Input: [B, seq_len, embed_dim]
        x = x + self.pos_embed
        # Output: [B, seq_len, embed_dim]
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        # Multi-head self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = residual + self.dropout1(attn_output)
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        return x

class VolumetricTransformer(nn.Module):
    def __init__(self, in_channels=256, embed_dim=512, depth=2, 
                 num_heads=8, mlp_ratio=4, dropout=0.1, patch_size=2):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.pos_encoding = PositionalEncoding3D(embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Convert to sequence
        x = self.patch_embedding(x)   # [B, seq_len, embed_dim]
        x = self.pos_encoding(x)      # Add positional information
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final normalization
        x = self.norm(x)
        
        return x

class TransformerProjection(nn.Module):
    def __init__(self, embed_dim=512, out_channels=256, spatial_dims=(4, 4, 4)):
        super().__init__()
        self.out_channels = out_channels
        self.h, self.w, self.d = spatial_dims
        
        # Project from embedding dimension to output channels
        self.projection = nn.Linear(embed_dim, out_channels)
        
    def forward(self, x):
        # Input: [B, seq_len, embed_dim]
        B, L, C = x.shape
        
        # Project to output channels
        x = self.projection(x)  # [B, seq_len, out_channels]
        
        # Reshape to spatial dimensions
        x = x.permute(0, 2, 1)  # [B, out_channels, seq_len]
        x = x.reshape(B, self.out_channels, self.h, self.w, self.d)
        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, query_dim, key_dim, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Normalization layers
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_key = nn.LayerNorm(key_dim)
        
        # Projections for query, key, value
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_dim, embed_dim)
        self.v_proj = nn.Linear(key_dim, embed_dim)
        
        # Output projection explicitly to query_dim
        self.out_proj = nn.Linear(embed_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_features, key_features):
        # Get shapes
        B, C_q, H, W, D = query_features.shape
        
        # Reshape to 2D
        q_2d = query_features.flatten(2).permute(0, 2, 1)  # [B, HWD, C_q]
        k_2d = key_features.flatten(2).permute(0, 2, 1)    # [B, HWD, C_k]
        
        # Layer normalization
        q_norm = self.norm_query(q_2d)
        k_norm = self.norm_key(k_2d)
        
        # Linear projections
        q = self.q_proj(q_norm)  # [B, HWD, embed_dim]
        k = self.k_proj(k_norm)  # [B, HWD, embed_dim]
        v = self.v_proj(k_norm)  # [B, HWD, embed_dim]
        
        # Multi-head attention
        # Reshape to multi-head form
        q = q.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, HWD, head_dim]
        k = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, HWD, head_dim]
        v = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, num_heads, HWD, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, num_heads, HWD, HWD]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)  # [B, num_heads, HWD, head_dim]
        
        # Reshape back
        out = out.permute(0, 2, 1, 3).contiguous().view(B, -1, self.embed_dim)  # [B, HWD, embed_dim]
        
        # Final projection
        out = self.out_proj(out)  # [B, HWD, query_dim]
        out = self.dropout(out)
        
        # Residual connection
        out = out + q_norm  # [B, HWD, query_dim]
        
        # Reshape back to original form
        out = out.permute(0, 2, 1).reshape(B, C_q, H, W, D)  # [B, C_q, H, W, D]
        
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.activation1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.activation2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.dropout = nn.Dropout3d(0.2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        
        x = self.dropout(x)
        return x

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x

class VoluFormerUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Parameters
        in_channels = config.IN_CHANNELS
        num_classes = config.NUM_CLASSES
        
        # Force specific architecture with fixed channels
        self.init_conv = InitialConvLayer(in_channels, 16)
        
        # Encoder path - fixed dimensions to match data flow
        self.encoder1 = EncoderStage(16, 32)         # [B, 16, 128, 128, 128] → [B, 32, 64, 64, 64]
        self.encoder2 = EncoderStage(32, 64)         # [B, 32, 64, 64, 64] → [B, 64, 32, 32, 32]
        self.encoder3 = EncoderStage(64, 128)        # [B, 64, 32, 32, 32] → [B, 128, 16, 16, 16]
        self.encoder4 = EncoderStage(128, 256)       # [B, 128, 16, 16, 16] → [B, 256, 8, 8, 8]
        
        # Volumetric transformer
        self.transformer = VolumetricTransformer(
            in_channels=256,              # Match bottleneck channels
            embed_dim=128,                # From config.TRANSFORMER_DIM
            depth=config.TRANSFORMER_LAYERS,
            num_heads=config.TRANSFORMER_HEADS,
            dropout=config.DROPOUT_RATE,
            patch_size=2
        )
        
        # Feature projection for transformer output
        self.transformer_projection = TransformerProjection(
            embed_dim=128,
            out_channels=256,             # Match bottleneck channels
            spatial_dims=(4, 4, 4)
        )
        
        # Bridge between encoder and decoder
        self.bridge = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )
        
        # Decoder path components
        # Upsampling operations
        self.upsample4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)  # [B, 256, 8, 8, 8] -> [B, 128, 16, 16, 16]
        self.upsample3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)   # [B, 128, 16, 16, 16] -> [B, 64, 32, 32, 32]
        self.upsample2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)    # [B, 64, 32, 32, 32] -> [B, 32, 64, 64, 64]
        self.upsample1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)    # [B, 32, 64, 64, 64] -> [B, 16, 128, 128, 128]
        
        # Fusion for transformer features
        self.fusion = CrossAttentionFusion(
            128, 256,                     # query_dim=decoder features (128), key_dim=transformer features (256)
            embed_dim=128,
            num_heads=config.TRANSFORMER_HEADS,
            dropout=config.DROPOUT_RATE
        )
        
        # Convolutional blocks after concatenation - CORRECT CHANNEL COUNTS
        self.decoder4_conv = ConvBlock(128 + 128, 128)   # Upsampled features (128) + skip connection (128)
        self.decoder3_conv = ConvBlock(64 + 64, 64)      # Upsampled features (64) + skip connection (64)
        self.decoder2_conv = ConvBlock(32 + 32, 32)      # Upsampled features (32) + skip connection (32)
        self.decoder1_conv = ConvBlock(16 + 16, 16)      # Upsampled features (16) + skip connection (16)
        
        # Final classification
        self.seg_head = SegmentationHead(16, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Weight initialization following ViT paper recommendations
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm3d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial features
        x0 = self.init_conv(x)           # [B, 4, 128, 128, 128] -> [B, 16, 128, 128, 128]
        
        # Encoder path with skip connections
        x1 = self.encoder1(x0)           # [B, 16, 128, 128, 128] -> [B, 32, 64, 64, 64]
        x2 = self.encoder2(x1)           # [B, 32, 64, 64, 64] -> [B, 64, 32, 32, 32]
        x3 = self.encoder3(x2)           # [B, 64, 32, 32, 32] -> [B, 128, 16, 16, 16]
        x4 = self.encoder4(x3)           # [B, 128, 16, 16, 16] -> [B, 256, 8, 8, 8]
        
        # Transformer branch
        transformer_features = self.transformer(x4)                # [B, 64, 128]
        
        # Project transformer features back to spatial domain
        trans_feat_spatial = self.transformer_projection(transformer_features)  # [B, 256, 4, 4, 4]
        
        # Bridge
        bridge_features = self.bridge(x4)                          # [B, 256, 8, 8, 8]
        
        # Decoder path with feature fusion
        # Stage 4
        x_up4 = self.upsample4(bridge_features)                    # [B, 128, 16, 16, 16]
        
        # Resize transformer features to match decoder features
        trans_feat_resized = F.interpolate(
            trans_feat_spatial, 
            size=x_up4.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )                                                          # [B, 256, 16, 16, 16]
        
        # Apply fusion
        x_up4_fused = self.fusion(x_up4, trans_feat_resized)       # [B, 128, 16, 16, 16]
        
        # Ensure skip connection has matching dimensions
        if x3.shape[2:] != x_up4_fused.shape[2:]:
            x3_resized = F.interpolate(
                x3, 
                size=x_up4_fused.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
        else:
            x3_resized = x3                                        # [B, 128, 16, 16, 16]
        
        # Concatenate and refine
        d4 = torch.cat([x_up4_fused, x3_resized], dim=1)           # [B, 128+128, 16, 16, 16]
        d4 = self.decoder4_conv(d4)                                # [B, 128, 16, 16, 16]
        
        # Stage 3
        x_up3 = self.upsample3(d4)                                 # [B, 64, 32, 32, 32]
        
        # Ensure skip connection has matching dimensions
        if x2.shape[2:] != x_up3.shape[2:]:
            x2_resized = F.interpolate(
                x2, 
                size=x_up3.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
        else:
            x2_resized = x2                                        # [B, 64, 32, 32, 32]
            
        d3 = torch.cat([x_up3, x2_resized], dim=1)                 # [B, 64+64, 32, 32, 32]
        d3 = self.decoder3_conv(d3)                                # [B, 64, 32, 32, 32]
        
        # Stage 2
        x_up2 = self.upsample2(d3)                                 # [B, 32, 64, 64, 64]
        
        # Ensure skip connection has matching dimensions
        if x1.shape[2:] != x_up2.shape[2:]:
            x1_resized = F.interpolate(
                x1, 
                size=x_up2.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
        else:
            x1_resized = x1                                        # [B, 32, 64, 64, 64]
            
        d2 = torch.cat([x_up2, x1_resized], dim=1)                 # [B, 32+32, 64, 64, 64]
        d2 = self.decoder2_conv(d2)                                # [B, 32, 64, 64, 64]
        
        # Stage 1
        x_up1 = self.upsample1(d2)                                 # [B, 16, 128, 128, 128]
        
        # Ensure skip connection has matching dimensions
        if x0.shape[2:] != x_up1.shape[2:]:
            x0_resized = F.interpolate(
                x0, 
                size=x_up1.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
        else:
            x0_resized = x0                                        # [B, 16, 128, 128, 128]
            
        d1 = torch.cat([x_up1, x0_resized], dim=1)                 # [B, 16+16, 128, 128, 128]
        d1 = self.decoder1_conv(d1)                                # [B, 16, 128, 128, 128]
        
        # Final segmentation
        logits = self.seg_head(d1)                                 # [B, 4, 128, 128, 128]
        
        return logits
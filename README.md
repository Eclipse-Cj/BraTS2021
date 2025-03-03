The first edition is another model. It doesn't perform well. Please ignore.

## Model Overview

SwinUTransBTS is a hybrid CNN-Transformer architecture designed for 3D brain tumor segmentation from multimodal MRI scans. This model combines the strengths of U-Net-like convolutional networks with Swin Transformer blocks to effectively capture both local and global features in volumetric medical images.

## Key Features

### Hybrid Architecture
- **Encoder-Decoder Structure**: U-Net-like architecture with skip connections for preserving spatial information
- **CNN Components**: Handle local feature extraction and provide efficient downsampling/upsampling
- **Swin Transformer Blocks**: Capture long-range dependencies between different regions

### Specialized Components
- **Shifted Window Attention**: Efficient self-attention within local windows with shifted windowing for cross-window connections
- **Key-Value Transformer Blocks**: Special transformer blocks in the decoder that use features from the encoder as keys and values
- **Multi-scale Processing**: Gradually reduces spatial dimensions while increasing feature channels

### Input/Output
- **Input**: 4-channel 3D MRI volumes (T1, T1ce, T2, FLAIR) of size 128×128×128
- **Output**: 4-class segmentation maps (Background, Necrotic Core/Non-Enhancing Tumor, Edema, Enhancing Tumor)

## Technical Implementation

The model processes 3D brain MRI scans through:

1. **Initial Convolution**: Transforms the 4-channel input into a feature representation
2. **Encoder Path**:
   - CNN blocks process and downsample the features
   - Swin Transformer blocks capture contextual information through self-attention
3. **Bottleneck**: Deepest part of the network with highest feature dimensionality
4. **Decoder Path**:
   - Upsampling blocks progressively recover spatial resolution
   - Skip connections from the encoder are integrated with specialized key-value transformer blocks
5. **Final Layer**: Maps features to 4-class segmentation output with softmax activation

## Data Processing Pipeline

- **Bias Field Correction**: N4ITK algorithm for intensity normalization
- **Intensity Normalization**: Z-score normalization of non-zero regions
- **Data Augmentation**: Random cropping, flipping, and intensity shifts
- **Sliding Window Inference**: Overlapping blocks during inference for consistent predictions

## Training Strategy

- **Combined Loss Function**: Weighted combination of Dice loss and Cross-Entropy loss
- **Gradient Accumulation**: To emulate larger batch sizes on memory-constrained GPUs
- **Learning Rate Scheduling**: Reduces learning rate when validation performance plateaus
- **Validation Strategy**: Fixed-blocks inference to monitor segmentation performance

## Performance Metrics

Evaluated on:
- **Dice Coefficient**: Overlap between predictions and ground truth
- **Sensitivity/Recall**: Model's ability to detect tumor regions
- **Specificity**: Model's ability to correctly identify non-tumor regions

Calculated separately for:
- **Whole Tumor (WT)**: All tumor tissues
- **Tumor Core (TC)**: Excluding edema
- **Enhancing Tumor (ET)**: Only the enhancing portions

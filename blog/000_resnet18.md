---
title: "ResNet18 Feature Map Viewer"
description: "An interactive visualization tool that allows users to explore feature maps of a pretrained ResNet-18 model layer by layer. Upload your own image or use examples to inspect channel-wise responses and overlays."
author: "Your Name"
date: "June 5, 2025"
tags: "Computer Vision, ResNet, Feature Visualization, Deep Learning"
demo_url: "https://huggingface.co/spaces/Zarkzh17/resnet18-viewer"
github_url: "https://github.com/zangzh17/resnet18-a1-clone"
# portfolio_url: "/index.html#portfolio"
home_url: "/index.html#blog"
---

## How it works

This demo extracts activations from a pretrained ResNet-18 model using PyTorch, visualizes individual channels as grayscale heatmaps, and overlays them on the input image. It uses Gradio Blocks as the backend (hosted on Hugging Face Spaces) and this page is deployed via GitHub Pages.

## Interactive Demo

Try the demo yourself below! You can upload your own images or use the provided examples to explore how different layers of the ResNet-18 model respond to various visual features.

<iframe src="https://Zarkzh17-resnet18-viewer.hf.space" width="100%" height="750" frameborder="0" style="border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); margin: 2rem 0;"></iframe>

## Technical Details

### Model Architecture

ResNet-18 is a convolutional neural network that consists of:

- **Initial Convolution**: 7×7 conv with 64 filters
- **Residual Blocks**: 4 groups of residual blocks with skip connections
- **Feature Maps**: Progressive reduction in spatial dimensions while increasing channels
- **Global Average Pooling**: Final feature aggregation before classification

## Understanding Feature Maps

### Early Layers (Conv1, Layer1)
Early layers typically detect:
- **Edges and lines** at various orientations
- **Simple textures** and basic patterns  
- **Color transitions** and gradients
- **Low-level geometric features**

### Middle Layers (Layer2, Layer3)
Middle layers capture:
- **Object parts** like corners, curves
- **Texture patterns** with more complexity
- **Shape combinations** and spatial relationships
- **Mid-level semantic features**

### Deep Layers (Layer4)
Deep layers represent:
- **High-level concepts** and object categories
- **Semantic features** related to specific classes
- **Complex patterns** that distinguish between objects
- **Abstract representations** for classification

## Code Example

Here's how you can extract and visualize feature maps programmatically:

```python
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

class FeatureExtractor:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        
        # Register hook to capture features
        self.hook = target_layer.register_forward_hook(self.save_features)
    
    def save_features(self, module, input, output):
        self.features = output.detach()
    
    def extract_features(self, x):
        _ = self.model(x)
        return self.features

# Load pretrained ResNet-18
model = models.resnet18(pretrained=True)
model.eval()

# Extract features from layer2
extractor = FeatureExtractor(model, model.layer2)
features = extractor.extract_features(input_tensor)

```

---

*Explore the interactive demo above to gain hands-on experience with how deep neural networks process and understand visual information!*
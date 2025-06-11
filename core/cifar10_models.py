"""
CIFAR-10 Neural Network Models

This module contains PyTorch model implementations for the CIFAR-10 dataset classification task.
It includes both fully connected (FC) and convolutional neural network (CNN) architectures.

Model definitions for CIFAR-10 classification using PyTorch.
Includes configurable FC and CNN architectures, and variants of ResNet and DenseNet.

The models are designed to be configurable and extensible, allowing for easy experimentation
with different architectures and hyperparameters.
"""

import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import torch
from torchvision import models

# ------------------------------------------------------------------------------------------------
# FC model
# ------------------------------------------------------------------------------------------------
class CIFAR10_FC(nn.Module):
    """
    Fully Connected Neural Network for CIFAR-10 classification.
    
    This model implements a configurable multi-layer perceptron with batch normalization,
    dropout, and customizable activation functions.
    
    Args:
        input_shape (Tuple[int, int, int]): Input shape (channels, height, width)
        num_classes (int): Number of output classes (10 for CIFAR-10)
        hidden_layers (List[int]): List of hidden layer sizes
        dropout_rates (List[float]): Dropout rates for each hidden layer
        activation_cls (nn.Module): Activation function class (e.g., nn.ReLU)
    """
    def __init__(
            self, 
            input_shape: Tuple[int, int, int], 
            num_classes: int, 
            hidden_layers: List[int], 
            dropout_rates: List[float], 
            activation_cls: nn.Module
        ):
        super(CIFAR10_FC, self).__init__()

        self.fl = nn.Flatten()
        layers = []
        prev_units = input_shape[0] * input_shape[1] * input_shape[2]
        self.input_shape = input_shape
        
        # Build hidden layers with batch norm, activation, and dropout
        for idx, units in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_units, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(activation_cls())
            layers.append(nn.Dropout(dropout_rates[idx]))
            prev_units = units

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.fl(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# ------------------------------------------------------------------------------------------------
# CNN model
# ------------------------------------------------------------------------------------------------
class CIFAR10_CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 classification.
    
    This model implements a configurable CNN architecture with customizable convolutional
    layers, fully connected layers, batch normalization, dropout, and activation functions.
    
    Args:
        input_shape (Tuple[int, int, int]): Input shape (channels, height, width)
        num_classes (int): Number of output classes (10 for CIFAR-10)
        conv_layers (List[dict]): List of convolutional layer configurations
        fc_layers (List[int]): List of fully connected layer sizes
        dropout_rates (Optional[List[float]]): Dropout rates for FC layers
        activation_cls (nn.Module): Activation function class (e.g., nn.ReLU)
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        conv_layers: List[dict],
        fc_layers: List[int],
        dropout_rates: Optional[List[float]] = None,
        activation_cls: nn.Module = nn.ReLU
    ):
        super().__init__()

        # Build convolutional feature extractor
        self.features = nn.Sequential()
        in_channels = input_shape[0]
        self.input_shape = input_shape

        for idx, layer_cfg in enumerate(conv_layers):
            out_channels = layer_cfg.get("out_channels", 32)
            kernel_size = layer_cfg.get("kernel_size", 3)
            stride = layer_cfg.get("stride", 1)
            padding = layer_cfg.get("padding", 1)
            pool = layer_cfg.get("pool", False)

            self.features.add_module(f"conv{idx}", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            self.features.add_module(f"bn{idx}", nn.BatchNorm2d(out_channels))
            self.features.add_module(f"act{idx}", activation_cls())
            if pool:
                self.features.add_module(f"pool{idx}", nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        # Compute flattened size for FC layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        assert fc_layers, "fc_layers must contain at least one layer"

        # Build classifier (FC layers)
        self.classifier = nn.Sequential()
        self.classifier.add_module("flatten", nn.Flatten())
        self.classifier.add_module("fc0", nn.Linear(flattened_size, fc_layers[0]))

        prev_units = fc_layers[0]
        for idx, units in enumerate(fc_layers[1:], 1):
            self.classifier.add_module(f"fc{idx}", nn.Linear(prev_units, units))
            self.classifier.add_module(f"bn{idx}", nn.BatchNorm1d(units))
            self.classifier.add_module(f"act{idx}", activation_cls())
            if dropout_rates and idx < len(dropout_rates):
                self.classifier.add_module(f"dropout{idx}", nn.Dropout(dropout_rates[idx]))
            prev_units = units

        self.classifier.add_module("output", nn.Linear(prev_units, num_classes))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------------------------------------------
# Advanced models
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Deep Dropout CNN
# ------------------------------------------------------------------------------------------------
class CIFAR10_DeepDropoutCNN(nn.Module):
    """
    Deep CNN for CIFAR-10 with progressively increasing dropout after each block.

    Args:
        input_shape (tuple): Input image shape (C, H, W)
        num_classes (int): Number of output classes
        conv_channels (List[int]): List of output channels per conv block
        dropout_schedule (List[float]): Dropout rate after each block
        activation_cls (nn.Module): Activation function class (e.g., nn.LeakyReLU)
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 126, 126),
        num_classes=10,
        conv_channels: List[int] = [320, 320, 640, 640, 960, 960, 1280, 1280, 1600, 1600, 1920, 1920],
        dropout_schedule: Optional[List[float]] = None,
        activation_cls: nn.Module = nn.LeakyReLU,
        use_batchnorm: bool = False
    ):
        super().__init__()
        C, H, W = input_shape

        if dropout_schedule is None:
            dropout_schedule = [0.1, 0.1, 0.2, 0.3, 0.4, 0.5]
        #print(f"Debug: Dropout schedule: {dropout_schedule}")
        #print(f"Debug: Conv channels: {conv_channels}")
        assert len(conv_channels) == len(dropout_schedule), "Dropout schedule must match number of conv layers"
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation_cls = activation_cls

        layers = []
        in_channels = C

        for i, (out_channels, dropout) in enumerate(zip(conv_channels, dropout_schedule)):
            kernel_size = 2 if i < len(conv_channels) - 2 else 1  # last two convs are C1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_cls(negative_slope=0.33))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            if (i + 1) % 2 == 0:
                layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        # Final layer to 10 classes
        layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))

        self.feature_extractor = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x)  # shape (B, C, 1, 1)
        return torch.flatten(x, 1)   # shape (B, num_classes)
    
# ------------------------------------------------------------------------------------------------
# ResNet18
# ------------------------------------------------------------------------------------------------
class CIFAR10_ResNet18(nn.Module):
    """ResNet18 backbone adapted for CIFAR-10."""

    def __init__(self, num_classes: int = 10, input_shape: Tuple[int, int, int] = (3, 32, 32)):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.input_shape = input_shape

    def forward(self, x):
        return self.model(x)

# ------------------------------------------------------------------------------------------------
# DenseNet121
# ------------------------------------------------------------------------------------------------
class CIFAR10_DenseNet121(nn.Module):
    """DenseNet121 backbone adapted for CIFAR-10."""

    def __init__(self, num_classes: int = 10, input_shape: Tuple[int, int, int] = (3, 32, 32)):
        super().__init__()
        self.model = models.densenet121(weights=None)
        self.model.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.features.pool0 = nn.Identity()
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, num_classes
        )
        self.input_shape = input_shape
        
    def forward(self, x):
        return self.model(x)
    
# ------------------------------------------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------------------------------------------
def create_resnet18_model(num_classes: int = 10, input_shape: Tuple[int, int, int] = (3, 32, 32), **kwargs):
    return CIFAR10_ResNet18(num_classes=num_classes, input_shape=input_shape)

def create_densenet121_model(num_classes: int = 10, input_shape: Tuple[int, int, int] = (3, 32, 32), **kwargs):
    return CIFAR10_DenseNet121(num_classes=num_classes, input_shape=input_shape)
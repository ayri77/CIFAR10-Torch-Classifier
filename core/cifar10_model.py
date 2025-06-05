"""
CIFAR-10 Neural Network Models

This module contains PyTorch model implementations for the CIFAR-10 dataset classification task.
It includes both fully connected (FC) and convolutional neural network (CNN) architectures.

The models are designed to be configurable and extensible, allowing for easy experimentation
with different architectures and hyperparameters.
"""

import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import torch

# ------------------------------------------------------------------------------------------------
# FC model
# ------------------------------------------------------------------------------------------------
class CIFAR10_FC(nn.Module):
    """
    Fully Connected Neural Network for CIFAR-10 classification.
    
    This model implements a configurable multi-layer perceptron with batch normalization,
    dropout, and customizable activation functions.
    
    Args:
        input_size (int): Size of the input features (e.g., 3072 for CIFAR-10)
        num_classes (int): Number of output classes (10 for CIFAR-10)
        hidden_layers (List[int]): List of hidden layer sizes
        dropout_rates (List[float]): Dropout rates for each hidden layer
        activation_cls (nn.Module): Activation function class (e.g., nn.ReLU)
    """
    def __init__(
            self, 
            input_size: int, 
            num_classes: int, 
            hidden_layers: List[int], 
            dropout_rates: List[float], 
            activation_cls: nn.Module
        ):
        super(CIFAR10_FC, self).__init__()

        self.fl = nn.Flatten()
        layers = []
        prev_units = input_size
        
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
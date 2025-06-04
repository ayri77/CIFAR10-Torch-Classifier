import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import torch

# ------------------------------------------------------------------------------------------------
# FC model
# ------------------------------------------------------------------------------------------------
class CIFAR10_FC(nn.Module):
    def __init__(
            self, 
            input_size, 
            num_classes, 
            hidden_layers, 
            dropout_rates, 
            activation_fn
        ):
                
        super(CIFAR10_FC, self).__init__()

        self.fl = nn.Flatten()
        layers = []
        prev_units = input_size
        for idx, units in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_units,units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rates[idx]))
            prev_units = units

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_units, num_classes)

    def forward(self, x):        
        x = self.fl(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# ------------------------------------------------------------------------------------------------
# CNN model
# ------------------------------------------------------------------------------------------------
class CIFAR10_CNN(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        conv_layers: List[dict],
        fc_layers: List[int],
        dropout_rates: Optional[List[float]] = None,
        activation_fn=nn.ReLU
    ):
        super().__init__()

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
            self.features.add_module(f"act{idx}", activation_fn())
            if pool:
                self.features.add_module(f"pool{idx}", nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        # Dummy forward to compute flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # e.g., (1, 3, 32, 32)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential()
        self.classifier.add_module("flatten", nn.Flatten())
        self.classifier.add_module("fc0", nn.Linear(flattened_size, fc_layers[0]))

        prev_units = fc_layers[0]
        for idx, units in enumerate(fc_layers[1:], 1):
            self.classifier.add_module(f"fc{idx}", nn.Linear(prev_units, units))
            self.classifier.add_module(f"bn{idx}", nn.BatchNorm1d(units))
            self.classifier.add_module(f"act{idx}", activation_fn())
            if dropout_rates:
                self.classifier.add_module(f"dropout{idx}", nn.Dropout(dropout_rates[idx]))
            prev_units = units

        self.classifier.add_module("output", nn.Linear(prev_units, num_classes))
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
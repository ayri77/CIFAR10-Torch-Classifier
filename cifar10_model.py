import torch.nn as nn
import config

# class
class CIFAR10_torch(nn.Module):
    def __init__(
            self, 
            input_size, 
            num_classes, 
            hidden_layers, 
            dropout_rate, 
            activation_fn
        ):
                
        super(CIFAR10_torch, self).__init__()

        self.fl = nn.Flatten()
        layers = []
        prev_units = input_size
        for idx, units in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_units,units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate[idx]))
            prev_units = units

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_units, num_classes)

    def forward(self, x):        
        x = self.fl(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
import torch.nn as nn

# class
class CIFAR10_torch(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers=[512, 256, 128], dropout_rate=[0.3,0.3,0.3], activation_fn = nn.ReLU):        
        super(CIFAR10_torch, self).__init__()

        self.hidden_layers = nn.ModuleList()
        prev_units = input_size
        self.fl = nn.Flatten()
        for idx, units in enumerate(hidden_layers):
            self.hidden_layers.append(nn.Linear(prev_units,units))
            self.hidden_layers.append(nn.BatchNorm1d(units))
            self.hidden_layers.append(activation_fn())
            self.hidden_layers.append(nn.Dropout(dropout_rate[idx]))
            prev_units = units

        self.output_layer = nn.Linear(prev_units, num_classes)

    def forward(self, x):        
        x = self.fl(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
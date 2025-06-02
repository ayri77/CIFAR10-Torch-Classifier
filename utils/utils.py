import random
import numpy as np
import torch
import json
import os
from cifar10_model import CIFAR10_FC, CIFAR10_CNN

def set_seed(seed=42):
    print(f"🧬 Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_deterministic(deterministic: bool = True, benchmark: bool = False):
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

def load_architecture(arch_name, base_dir="architectures"):

    model_class_mapping = {
        "CIFAR10_CNN": CIFAR10_CNN,
        "CIFAR10_FC": CIFAR10_FC
    }

    path = os.path.join(base_dir, f"{arch_name}.json")
    with open(path, 'r') as f:
        arch_data = json.load(f)

    model_type = arch_data.pop("model_type")
    model_class = model_class_mapping[model_type]

    return model_class, arch_data

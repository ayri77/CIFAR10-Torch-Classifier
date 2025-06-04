from utils.paths import MODELS_DIR, DATA_DIR, ARCHITECTURES_DIR

import random
import numpy as np
import torch
import json
import os
from core.cifar10_model import CIFAR10_FC, CIFAR10_CNN
from config import AUGMENTATION, GRAYSCALE

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

def load_architecture(arch_name, base_dir=ARCHITECTURES_DIR):
    import torch.nn as nn

    model_class_mapping = {
        "CIFAR10_CNN": CIFAR10_CNN,
        "CIFAR10_FC": CIFAR10_FC
    }

    path = os.path.join(base_dir, f"{arch_name}.json")
    with open(path, 'r') as f:
        arch_data = json.load(f)

    # extract and pop values
    model_type = arch_data.pop("model_type")
    model_class = model_class_mapping[model_type]

    # activation_fn as string → keep in separate variable
    activation_fn_name = arch_data.pop("activation_fn", "ReLU")

    # optimizer config
    optimizer_config = arch_data.pop("optimizer", {
        "name": "Adam",
        "kwargs": {"lr": 0.001}
    })

    # criterion config
    criterion_config = arch_data.pop("criterion", {
        "name": "CrossEntropyLoss",
        "kwargs": {}
    })

    # lr_scheduler config
    lr_scheduler_config = arch_data.pop("lr_scheduler", None)

    # augmentation config
    augmentation = arch_data.pop("augmentation", AUGMENTATION)
    grayscale = arch_data.pop("grayscale", GRAYSCALE)

    return model_class, arch_data, activation_fn_name, optimizer_config, criterion_config, lr_scheduler_config, augmentation, grayscale
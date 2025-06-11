"""
utils.py

Utility functions for reproducibility and loading model architectures in the CIFAR-10 classification project.

This module includes:
- Functions for setting the seed for the random number generators
- Functions for setting the deterministic flag for the cuDNN library
- Functions for loading the architecture from the JSON file
"""

from utils.paths import ARCHITECTURES_DIR

import random
import numpy as np
import torch
import json
import os
from config import AUGMENTATION, GRAYSCALE, RESIZE


def set_seed(seed=42, verbose=True):
    """
    Set the seed for the random number generators.
    Args:
        seed (int): The seed to set.
        verbose (bool): Whether to print the seed.
    """
    if verbose:
        print(f"🔧 Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_deterministic(deterministic: bool = True, benchmark: bool = False):
    """
    Set the deterministic flag for the cuDNN library.
    """
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def load_architecture(arch_name, base_dir=ARCHITECTURES_DIR):
    """
    Loads model architecture and configuration from a JSON file.

    Args:
        arch_name (str): Name of the architecture file (without extension)
        base_dir (str): Path to the directory containing architecture files

    Returns:
        tuple: (model_class, model_kwargs, activation_fn_name,
                optimizer_cfg, criterion_cfg, lr_scheduler_cfg,
                augmentation, grayscale)
    """ 

    path = os.path.join(base_dir, f"{arch_name}.json")
    with open(path, "r") as f:
        arch_data = json.load(f)

    # extract and pop values
    model_type = arch_data.pop("model_type")
    model_kwargs = arch_data.pop("model_kwargs", {})

    # optimizer config
    optimizer_config = arch_data.pop(
        "optimizer", {"name": "Adam", "kwargs": {"lr": 0.001}}
    )

    # criterion config
    criterion_config = arch_data.pop(
        "criterion", {"name": "CrossEntropyLoss", "kwargs": {}}
    )

    # lr_scheduler config
    lr_scheduler_config = arch_data.pop("lr_scheduler", None)

    # augmentation config
    augmentation = arch_data.pop("augmentation", AUGMENTATION)
    grayscale = arch_data.pop("grayscale", GRAYSCALE)
    resize = arch_data.pop("resize", RESIZE)

    return (
        model_type,
        model_kwargs,
        optimizer_config,
        criterion_config,
        lr_scheduler_config,
        augmentation,
        grayscale,
        resize,
    )

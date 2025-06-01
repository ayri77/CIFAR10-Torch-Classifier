import random
import numpy as np
import torch

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
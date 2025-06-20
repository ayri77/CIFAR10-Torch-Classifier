"""
Global configuration for CIFAR-10 training pipeline.
Contains settings for data processing, model architecture, training parameters,
device configuration, and logging.
"""

import torch

# ------------------------------------------------------------------------------------------------
# Random seed for reproducibility
# ------------------------------------------------------------------------------------------------
SEED = 42
# Reproducibility
DETERMINISTIC = True
BENCHMARK = False

# ------------------------------------------------------------------------------------------------
# Data loading and preprocessing
# ------------------------------------------------------------------------------------------------

# Data
SPLIT_RATIO = 0.8

# Data loaders
MEAN = [0.4913996756076813, 0.48215851187705994, 0.4465310275554657]
STD = [0.24703219532966614, 0.24348489940166473, 0.2615877091884613]

# Transforms
AUGMENTATION = {
    "mode": "basic",
    "mixup_alpha": 0.4,
    "cutout_size": 8
}
GRAYSCALE = False
RESIZE = (32, 32)

# ------------------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------------------
INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 10

HIDDEN_LAYERS = [512, 256, 128]
DROPOUT_RATES = [0.3, 0.3, 0.3]
ACTIVATION_FN = "ReLU"

# ------------------------------------------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------------------------------------------
NUM_EPOCHS = 100
# Patience for early stopping
PATIENCE = 10 
LR = 0.01
OPTIMIZER = "Adam"  # Alternatives: "SGD", "AdamW", "RMSprop", "Adagrad"
OPTIMIZER_KWARGS = {"lr": LR} # SGD/RMSprop: {"lr": LR, "momentum": 0.9}
CRITERION = "CrossEntropyLoss"
CRITERION_KWARGS = {} # {"weight": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])}

# StepLR example config (not default)
LR_SCHEDULER = "StepLR" # Alternatives: "ReduceLROnPlateau"
LR_SCHEDULER_KWARGS = {"step_size": 10, "gamma": 0.5}
# Alternatives: {"mode": "min", "factor": 0.5, "patience": 3, "threshold": 1e-4, "cooldown": 0, "min_lr": 1e-6, "verbose": True}

# ------------------------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------------------------
LOG_TENSORBOARD = True
LOG_DIR = "runs"
MODEL_DIR = "models"

# ------------------------------------------------------------------------------------------------
# Device and batch size
# ------------------------------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()
NUM_WORKERS = 1 # os.cpu_count()
BATCH_SIZE = 24
PIN_MEMORY = True

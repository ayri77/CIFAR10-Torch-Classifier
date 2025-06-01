import torch
import os

# config.py
SEED = 42

# Data
SPLIT_RATIO = 0.8

# Data loaders
MEAN = [0.4913996756076813, 0.48215851187705994, 0.4465310275554657]
STD = [0.24703219532966614, 0.24348489940166473, 0.2615877091884613]

# Model
INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 10
HIDDEN_LAYERS = [512, 256, 128]
DROPOUT_RATES = [0.3, 0.3, 0.3]
ACTIVATION_FN = "ReLU"

# Training
NUM_EPOCHS = 100
PATIENCE = 10
LR = 0.01
OPTIMIZER = "Adam" # "SGD"
OPTIMIZER_KWARGS = {"lr": LR} # {"lr": LR, "momentum": 0.9}
CRITERION = "CrossEntropyLoss"
CRITERION_KWARGS = {} # {"weight": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])}

# Logging
LOG_TENSORBOARD = True
LOG_DIR = "runs"
MODEL_DIR = "models"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 256
PIN_MEMORY = True

# Reproducibility
DETERMINISTIC = True
BENCHMARK = False

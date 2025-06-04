import os

# Project root = this file's grandparent directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(project_root, "data")
MODELS_DIR = os.path.join(project_root, "models")
ARCHITECTURES_DIR = os.path.join(project_root, "architectures")

# Tensorboard
TENSORBOARD_DIR = os.path.join(project_root, "runs")

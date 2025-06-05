"""
Path configuration for the CIFAR-10 project.

This module defines absolute paths for key project directories,
based on the location of the current file.
"""

import os

# Project root = this file's grandparent directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Main directories
DATA_DIR = os.path.join(project_root, "data")
MODELS_DIR = os.path.join(project_root, "models")
ARCHITECTURES_DIR = os.path.join(project_root, "architectures")

# TensorBoard logs
TENSORBOARD_DIR = os.path.join(project_root, "runs")

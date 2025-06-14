"""
train.py — Train CIFAR-10 model using CLI parameters.

Usage examples:
---------------
# Train with default config
python train.py --config architectures/cnn_model.json

# Train with grayscale input and augmentation
python train.py --config architectures/cnn_model.json --grayscale True --augmentation True

# Override learning rate and batch size
python train.py --config architectures/cnn_model.json --lr 0.001 --batch_size 128

Arguments:
----------
--config           Path to model architecture .json file (required)
--lr               Override learning rate from config (optional)
--epochs           Number of training epochs
--batch_size       Batch size for training
--patience         Patience for early stopping
--device           Device to use (cuda or cpu)
--log_tensorboard  Enable TensorBoard logging
--early_stopping   Use early stopping or not
--verbose          Verbose output
--save_model       Save model after training
--save_path        Directory to save model and metrics
--num_workers      Number of DataLoader workers
--augmentation     Enable data augmentation (overrides config)
--grayscale        Convert input to grayscale
--mixup_enabled    Enable mixup (overrides config)
--mixup_alpha      Mixup alpha (overrides config)
"""

# Built-in
import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import torch
from torchvision import transforms

from utils.utils import load_architecture
from config import MEAN, STD, NUM_CLASSES, SPLIT_RATIO, NUM_WORKERS, PATIENCE, NUM_EPOCHS, BATCH_SIZE
from utils.paths import MODELS_DIR, DATA_DIR, ARCHITECTURES_DIR
from core.cifar10_classifier import CIFAR10Classifier
from utils.data_utils import (
    compute_mean_std, get_transforms,
    load_cifar10_datasets, split_train_val,
    create_loaders, get_dataset_info
)

# Environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def override(cli_val, json_val):
    '''
    Override the value from the JSON file with the value from the CLI.
    Args:
        cli_val (any): The value from the CLI.
        json_val (any): The value from the JSON file.
    Returns:
        any: The overridden value.
    '''
    return cli_val if cli_val is not None else json_val

def parse_args():
    '''
    Parse the arguments from the CLI.
    Returns:
        argparse.Namespace: The parsed arguments.
    '''
    parser = argparse.ArgumentParser(description="Train CIFAR-10 model via CLI")

    parser.add_argument('--config', type=str, help="Path to architecture config")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--patience', type=int, default=PATIENCE, help="Patience")
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument('--log_tensorboard', type=bool, default=False, help="Log tensorboard")
    parser.add_argument('--early_stopping', type=bool, default=True, help="Early stopping")
    parser.add_argument('--verbose', type=bool, default=True, help="Verbose")
    parser.add_argument('--save_model', type=bool, default=True, help="Save model")
    parser.add_argument('--save_path', type=str, default="models", help="Save path")
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help="Number of workers")


    return parser.parse_args()

def main():
    '''
    Main function.
    '''
    args = parse_args()    

    # Step 0: Load architecture and model class
    (
        model_type, model_kwargs, optimizer_cfg, criterion_cfg, lr_scheduler_cfg, 
        augmentation, grayscale, resize
    ) = load_architecture(args.config)


    # Step 1: Load raw training data (no normalization)
    if grayscale:
        transform_gs = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])    
    raw_dataset, _ = load_cifar10_datasets(
        data_dir=DATA_DIR,
        transform=transform_gs if grayscale else transforms.ToTensor(),
        subset="train"
    )

    # Step 2: Dataset info
    input_shape, num_classes = get_dataset_info(raw_dataset)

    # Step 3: Compute mean and std
    mean, std = compute_mean_std(raw_dataset)

    # Step 4: Build model    
    model_cls = CIFAR10Classifier(
        name=args.config,
        model_type=model_type,
        model_kwargs=model_kwargs,
        input_shape=input_shape,
        num_classes=num_classes,        
        optimizer_name=optimizer_cfg["name"],
        optimizer_kwargs=optimizer_cfg["kwargs"],
        criterion_name=criterion_cfg["name"],
        criterion_kwargs=criterion_cfg["kwargs"],
        lr_scheduler_name=lr_scheduler_cfg["name"],
        lr_scheduler_kwargs=lr_scheduler_cfg["kwargs"],
        device=torch.device(args.device),
        mean=mean.tolist(),
        std=std.tolist(),
        augmentation=augmentation,
        grayscale=grayscale,
        resize=resize
    )

    # Step 5: Build transform and load full training set
    train_transform = get_transforms(
        mean=mean,
        std=std,
        augmentation=augmentation,
        grayscale=grayscale,
        resize=resize
    )    
    train_dataset, _ = load_cifar10_datasets(
        data_dir=DATA_DIR, 
        transform=train_transform, 
        subset="train"
    )

    # Step 6: Train/val split
    train_subset, val_subset = split_train_val(train_dataset, split_ratio=SPLIT_RATIO)

    # Step 7: Loaders
    train_loader, val_loader, _ = create_loaders(
        train_subset=train_subset,
        val_subset=val_subset,
        test_dataset=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Step 8: Build model
    model_cls.build_model()
    model_cls.compile()
    model_cls.summary()

    # Step 9: Train
    model_cls.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        patience=args.patience,
        log_tensorboard=args.log_tensorboard
    )

    print(f"\n✅ Training completed for model: {model_cls.name}")

if __name__ == "__main__":
    main()

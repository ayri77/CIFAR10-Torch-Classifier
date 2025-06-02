import argparse

import torch
from torchvision import transforms

from utils import load_architecture
from config import MEAN, STD, NUM_CLASSES, SPLIT_RATIO, NUM_WORKERS, PATIENCE, NUM_EPOCHS, BATCH_SIZE
from cifar10_classifier import CIFAR10Classifier
from utils import (
    compute_mean_std, get_transforms,
    load_cifar10_datasets, split_train_val,
    create_loaders, get_dataset_info
)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 model via CLI")

    parser.add_argument('--config', type=str, help="Path to architecture config (.json)")
    parser.add_argument('--lr', type=float, help="Learning rate override")
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
    args = parse_args()

    # Load architecture and model class
    model_class, model_kwargs = load_architecture(args.config)

    # Step 1: Load raw training data (no normalization)
    raw_dataset, _ = load_cifar10_datasets(transform=transforms.ToTensor(), subset="train")

    # Step 2: Compute mean and std
    mean, std = compute_mean_std(raw_dataset)

    # Step 3: Build transform and load full training set
    full_transform = get_transforms(mean, std)
    train_dataset, _ = load_cifar10_datasets(transform=full_transform, subset="train")

    # Step 4: Train/val split
    train_subset, val_subset = split_train_val(train_dataset, split_ratio=SPLIT_RATIO)

    # Step 5: Loaders
    train_loader, val_loader, _ = create_loaders(
        train_subset=train_subset,
        val_subset=val_subset,
        test_dataset=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Step 6: Dataset info
    input_shape, num_classes = get_dataset_info(train_dataset)

    # Build model
    model_cls = CIFAR10Classifier(
        name=args.config.split("/")[-1].replace(".json", ""),
        model_class=model_class,
        model_kwargs=model_kwargs,
        input_shape=input_shape,
        num_classes=num_classes,
        activation_fn_name="ReLU",
        optimizer_name="Adam",
        optimizer_kwargs={"lr": args.lr or 0.0005},
        criterion_name="CrossEntropyLoss",
        criterion_kwargs={},
        device=torch.device(args.device),
        mean=MEAN,
        std=STD
    )

    model_cls.build_model()
    model_cls.compile()
    model_cls.summary()

    # Train
    model_cls.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        patience=args.patience,
        log_tensorboard=False
    )

if __name__ == "__main__":
    main()

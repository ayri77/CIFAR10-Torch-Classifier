import torch
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config

def compute_mean_std(dataset, batch_size=512):
    print("📊 Computing mean and std...")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sum_ = torch.zeros(3)
    squared_sum = torch.zeros(3)
    num_batches = 0
    for data, _ in loader:
        sum_ += data.sum(dim=[0, 2, 3])
        squared_sum += (data ** 2).sum(dim=[0, 2, 3])
        num_batches += data.size(0) * data.size(2) * data.size(3)
    mean = sum_ / num_batches
    std = (squared_sum / num_batches - mean ** 2).sqrt()
    print(f"✅ Mean: {mean.tolist()}, Std: {std.tolist()}")
    return mean, std


def get_transforms(mean, std):
    print("🧪 Creating normalization transform...")
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])


def load_cifar10_datasets(data_dir="./data", transform=None, subset="full"):
    print(f"📥 Downloading/loading CIFAR-10 datasets... Loading {subset} dataset")
    if subset == "full":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    elif subset == "train":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = None
    elif subset == "test":
        train_dataset = None
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    if train_dataset is not None:
        print(f"✅ Loaded training samples: {len(train_dataset)}")
    if test_dataset is not None:
        print(f"✅ Loaded test samples: {len(test_dataset)}")
    return train_dataset, test_dataset


def split_train_val(train_dataset, split_ratio=0.8):
    print(f"🔀 Splitting dataset with ratio {split_ratio:.2f}...")
    total = len(train_dataset)
    train_len = int(total * split_ratio)
    val_len = total - train_len
    print(f"✅ Train size: {train_len}, Validation size: {val_len}")
    return random_split(train_dataset, [train_len, val_len])


def create_loaders(train_subset=None, val_subset=None, test_dataset=None, batch_size=64, num_workers=4):
    print(f"📦 Creating data loaders with batch size {batch_size}...")
    if train_subset is None:
        train_loader = None
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=seed_worker,
        prefetch_factor=2
        )

    if val_subset is None:
        val_loader = None
    else:
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.PIN_MEMORY,
        worker_init_fn=seed_worker,
        prefetch_factor=2
        )

    if test_dataset is None:
        test_loader = None
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=config.PIN_MEMORY,
        worker_init_fn=seed_worker,
        prefetch_factor=2
    )
    print("✅ Data loaders ready.")
    return train_loader, val_loader, test_loader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset_info(dataset):
    print("🔍 Extracting dataset info...")
    image, _ = dataset[0]
    input_shape = image.shape  # (C, H, W)
    num_classes = len(dataset.classes)
    print(f"✅ Input shape: {input_shape}, Number of classes: {num_classes}")
    return input_shape, num_classes

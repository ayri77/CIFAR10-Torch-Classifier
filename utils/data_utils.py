import torch
import numpy as np
import random
import os
import sys
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config

from cifar10_classifier import CIFAR10Classifier

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

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
        transforms.Resize((32, 32)),
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

# ------------------------------------------------------------------------------------------------
# Models comparing
# ------------------------------------------------------------------------------------------------
def evaluate_all_models_on_test(models_dir="models", force=False, save_predictions=True):
    results = {}
    class_names_path = os.path.join(project_root, "data", "class_names.json")
    class_names = None
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = json.load(f)

    for model_name in os.listdir(models_dir):
        print("-"*60)
        print(f"Evaluating {model_name}...")
        print("-"*60)        
        model_path = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        test_result_path = os.path.join(model_path, f"{model_name}_test.json")
        config_path = os.path.join(model_path, f"{model_name}_config.json")
        model_file = os.path.join(model_path, f"{model_name}_best_model.pth")

        if not os.path.exists(config_path) or not os.path.exists(model_file):
            continue

        if not force and os.path.exists(test_result_path):
            with open(test_result_path, "r") as f:
                results[model_name] = json.load(f)
            print(f"✅ Test results already exist for {model_name}.. loading from {test_result_path}")
            continue

        model = CIFAR10Classifier.load_model(
            model_name=model_name,
            config_path=config_path,
            model_path=model_file
        )

        # prepare test loader
        mean, std = torch.tensor(model.mean), torch.tensor(model.std)
        test_transform = get_transforms(mean, std)
        data_dir = os.path.join(project_root, "data")
        _, test_dataset = load_cifar10_datasets(data_dir=data_dir, transform=test_transform, subset="test")
        _, _, test_loader = create_loaders(
            test_dataset=test_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=1
        )
        class_names = test_dataset.classes        
        with open(class_names_path, "w") as f:
            json.dump(class_names, f)        

        metrics = model.evaluate(test_loader, verbose=False)

        # optionally drop large arrays if we don’t need them cached
        if not save_predictions:
            metrics.pop("y_pred", None)
            metrics.pop("y_true", None)

        with open(test_result_path, "w") as f:
            json.dump(metrics, f, indent=4)
            print(f"✅ Test metrics saved to {test_result_path}")

        results[model_name] = metrics

    return results, class_names
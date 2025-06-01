import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision

from collections import Counter

def show_random_samples(dataset, class_names, n=10):
    """
    Show n random images from the dataset with their labels.

    Args:
        dataset (torch.utils.data.Dataset): CIFAR10 dataset
        class_names (list): List of class names
        n (int): Number of images
    """
    indices = random.sample(range(len(dataset)), n)
    images_labels = [dataset[i] for i in indices]
    
    plt.figure(figsize=(15, 2))
    for i, (image, label) in enumerate(images_labels):
        image_np = image.permute(1, 2, 0).numpy()
        mean = np.array([0.0, 0.0, 0.0])
        std = np.array([1.0, 1.0, 1.0])
        if hasattr(dataset, 'transform') and dataset.transform is not None:
            # If there is a Normalize, try to reverse it
            for t in dataset.transform.transforms:
                if isinstance(t, torchvision.transforms.Normalize):
                    mean = np.array(t.mean)
                    std = np.array(t.std)
        image_np = std * image_np + mean  # de-normalization
        image_np = np.clip(image_np, 0, 1)

        plt.subplot(1, n, i + 1)
        plt.imshow(image_np)
        plt.title(class_names[label])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_class_distribution(dataset, class_names):
    """
    Show the number of examples for each class in the dataset.

    Args:
        dataset: CIFAR10 or Subset
        class_names: list of class names
    """
    targets = []

    # Subset or Dataset
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    elif hasattr(dataset, 'targets'):
        targets = dataset.targets

    counter = Counter(targets)
    labels = [class_names[i] for i in range(len(class_names))]
    counts = [counter[i] for i in range(len(class_names))]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, counts, color='skyblue')
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_training_history(history, save_path=None):
    """
    Plots training and validation loss and accuracy from history list.

    Args:
        history (List[Dict]): Training log containing epoch-level metrics.
        save_path (str, optional): Path to save the figure. Defaults to None.
    """
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Training plot saved to {save_path}")
    else:
        plt.show()
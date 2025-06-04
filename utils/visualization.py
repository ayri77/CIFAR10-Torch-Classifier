from utils.paths import MODELS_DIR, DATA_DIR, ARCHITECTURES_DIR

import numpy as np
import random
import torchvision
import os
import json

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

from collections import Counter

def show_random_samples(dataset_raw, dataset_aug, class_names, n=10):
    """
    Show n random images from the dataset with their labels.

    Args:
        dataset_raw (torch.utils.data.Dataset): CIFAR10 dataset (raw)
        dataset_aug (torch.utils.data.Dataset): CIFAR10 dataset (augmented)
        class_names (list): List of class names
        n (int): Number of images
    """
    indices = random.sample(range(len(dataset_raw)), n)

    plt.figure(figsize=(n * 2, 4))

    for i, idx in enumerate(indices):
        # Original image
        image_raw, label = dataset_raw[idx]
        plt.subplot(2, n, i + 1)
        img_raw_np = image_raw.permute(1, 2, 0).numpy()
        img_raw_np = (img_raw_np - img_raw_np.min()) / (img_raw_np.max() - img_raw_np.min())
        plt.imshow(img_raw_np)
        title = f"{class_names[label]}" if class_names else f"Class {label}"
        plt.title(f"Orig: {title}", fontsize=8)
        plt.axis("off")

        plt.subplot(2, n, n + i + 1)
        # Augmented image
        image_aug, _ = dataset_aug[idx]        
        # Convert tensor to numpy image
        img_aug_np = image_aug.numpy()
        # Check for grayscale image
        if img_aug_np.shape[0] == 1:
            img_aug_np = img_aug_np.squeeze(0)  # shape: [H, W]
            plt.imshow(img_aug_np, cmap='gray')
        else:
            img_aug_np = np.transpose(img_aug_np, (1, 2, 0))  # shape: [H, W, C]
            img_aug_np = (img_aug_np - img_aug_np.min()) / (img_aug_np.max() - img_aug_np.min())
            plt.imshow(img_aug_np)
        
        plt.title(f"Augm: {title}", fontsize=8)
        plt.axis("off")

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

def plot_confusion_matrix(y_pred_classes, y_true, class_names=None, normalize=False, model_name=None):

    cm = confusion_matrix(y_true, y_pred_classes, normalize='true' if normalize else None)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {model_name}")
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

# ------------------------------------------------------------------------------------------------
# Models comparing
# ------------------------------------------------------------------------------------------------

def build_comparison_table(models_dir=MODELS_DIR):
    rows = []

    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        metrics_file = os.path.join(model_path, f"{model_name}_metrics.json")
        config_file = os.path.join(model_path, f"{model_name}_config.json")

        if not os.path.exists(metrics_file) or not os.path.exists(config_file):
            continue

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        with open(config_file, "r") as f:
            config = json.load(f)

        best_epoch = max(metrics, key=lambda e: e["val_accuracy"])
        last_epochs = metrics[-5:]

        # Determine model type
        if "conv_layers" in config:
            model_type = "CNN"
            layer_summary = f"conv: {len(config['conv_layers'])}, fc: {len(config['fc_layers'])}"
        else:
            model_type = "FC"
            layer_summary = f"fc: {len(config['hidden_layers'])}"

        overfitting_gap = round(best_epoch["train_accuracy"] - best_epoch["val_accuracy"], 4)
        val_stability = np.std([e["val_accuracy"] for e in last_epochs])
        avg_epoch_time = round(np.mean([e["epoch_time"] for e in metrics]), 2)

        # Estimate convergence epoch: first epoch reaching 90% of best val acc
        threshold = 0.9 * best_epoch["val_accuracy"]
        epochs_to_threshold = next((e["epoch"] for e in metrics if e["val_accuracy"] >= threshold), None)

        rows.append({
            "Model": model_name,
            "Type": model_type,
            "Architecture": layer_summary,
            "Epoch (best)": best_epoch["epoch"],
            "Train Acc": round(best_epoch["train_accuracy"], 4),
            "Val Acc": round(best_epoch["val_accuracy"], 4),
            "Overfit Gap": overfitting_gap,
            "Val Loss": round(best_epoch["val_loss"], 4),
            "Avg Epoch Time (s)": avg_epoch_time,
            "LR": config.get("optimizer_kwargs", {}).get("lr", None),
            "Dropout": str(config.get("dropout_rates", "N/A")),
            "Optimizer": config.get("optimizer", "N/A"),
            "Converged by Epoch": epochs_to_threshold,
            "Stability (val acc)": round(val_stability, 4)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Val Acc", ascending=False).reset_index(drop=True)
    return df

def plot_model_comparison(df):
    import matplotlib.pyplot as plt
    df = df.sort_values("Val Acc", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(df["Model"], df["Val Acc"])
    plt.xticks(rotation=45)
    plt.ylabel("Validation Accuracy")
    plt.title("Model Comparison by Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_val_vs_test_acc(df):
    df_sorted = df.sort_values("Val Acc", ascending=False)
    plt.figure(figsize=(10, 5))
    x = range(len(df_sorted))

    plt.bar(x, df_sorted["Val Acc"], label="Val Acc", alpha=0.7)
    plt.bar(x, df_sorted["Test Acc"], label="Test Acc", alpha=0.7)
    plt.xticks(x, df_sorted["Model"], rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Validation vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_dropout_vs_overfit(df):
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Dropout Avg"], df["Overfit Gap"], s=100, alpha=0.7)
    for i, row in df.iterrows():
        plt.text(row["Dropout Avg"] + 0.005, row["Overfit Gap"], row["Model"], fontsize=8)
    plt.xlabel("Average Dropout")
    plt.ylabel("Overfitting Gap (Train Acc - Val Acc)")
    plt.title("Dropout Rate vs Overfitting Gap")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_efficiency(df):
    df_sorted = df.sort_values("Efficiency", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(df_sorted["Model"], df_sorted["Efficiency"], color="mediumseagreen")
    plt.xticks(rotation=45)
    plt.ylabel("Val Acc / Epoch Time")
    plt.title("Model Efficiency (Accuracy per Second)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_group_accuracy(df):
    melted = df.melt(
        id_vars=["Model", "Type"],
        value_vars=["Val Acc", "Test Acc"],
        var_name="Metric",
        value_name="Accuracy"
    )

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=melted, x="Type", y="Accuracy", hue="Metric")
    plt.title("Model Accuracy Distribution by Architecture Type")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_per_class_accuracy(y_true, y_pred, class_names=None):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
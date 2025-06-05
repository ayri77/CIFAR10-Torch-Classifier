"""
visualization.py

Visualization utilities for training analysis, evaluation metrics, and model comparison in the CIFAR-10 project.

This module includes:
- Functions for showing random samples from the dataset
- Functions for showing the class distribution
- Functions for plotting the confusion matrix
- Functions for plotting the training history
- Functions for comparing models
- Functions for plotting the model comparison
- Functions for plotting the dropout vs overfit
- Functions for plotting the efficiency
- Functions for plotting the group accuracy
"""

# Built-in
import os
import json
import random
from collections import Counter

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import display, clear_output
import ipywidgets as widgets

# Local
from utils.paths import MODELS_DIR

def show_random_samples(dataset_raw, dataset_aug, class_names, n=10):
    '''
    Show n random images from the dataset with their labels.    
    Args:
        dataset_raw (torch.utils.data.Dataset): CIFAR10 dataset (raw)
        dataset_aug (torch.utils.data.Dataset): CIFAR10 dataset (augmented)
        class_names (list): List of class names
        n (int): Number of images
    '''
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
    '''
    Show the number of examples for each class in the dataset.

    Args:
        dataset: CIFAR10 or Subset
        class_names: list of class names
    '''
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
    '''
    Plot the confusion matrix.
    Args:
        y_pred_classes (list): The predicted classes.
        y_true (list): The true classes.
        class_names (list): The class names.
        normalize (bool): Whether to normalize the confusion matrix.
        model_name (str): The name of the model.
    '''
    cm = confusion_matrix(y_true, y_pred_classes, normalize='true' if normalize else None)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 9})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()
    plt.close()

def plot_training_history(history, save_path=None):
    '''
    Plots training and validation loss and accuracy from history list.

    Args:
        history (List[Dict]): Training log containing epoch-level metrics.
        save_path (str, optional): Path to save the figure. Defaults to None.
    '''
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
    '''
    Build the comparison table.
    Args:
        models_dir (str): The directory to save the models.
    '''
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
        val_stability = round(np.std([e["val_accuracy"] for e in last_epochs]), 6)
        avg_epoch_time = round(np.mean([e["epoch_time"] for e in metrics]), 2)

        threshold = 0.9 * best_epoch["val_accuracy"]
        epochs_to_threshold = next((e["epoch"] for e in metrics if e["val_accuracy"] >= threshold), None)

        lr = config.get("optimizer_kwargs", {}).get("lr", None)
        dropout = config.get("dropout_rates", "N/A")
        if isinstance(dropout, list):
            dropout = [round(d, 2) for d in dropout]

        rows.append({
            # 🧱 Model Setup
            "Model": model_name,
            "Type": model_type,
            "Architecture": layer_summary,
            "Optimizer": config.get("optimizer", "N/A"),
            "Dropout": str(dropout),
            "LR": round(lr, 6) if lr else None,

            # 📊 Learning Metrics
            "Epoch (best)": best_epoch["epoch"],
            "Train Acc": round(best_epoch["train_accuracy"], 4),
            "Val Acc": round(best_epoch["val_accuracy"], 4),
            "Val Loss": round(best_epoch["val_loss"], 4),
            "Overfit Gap": overfitting_gap,
            "Avg Epoch Time (s)": avg_epoch_time,
            "Converged by Epoch": epochs_to_threshold,
            "Stability (val acc)": val_stability,
        })

    # Final dataframe
    df = pd.DataFrame(rows)

    # Preferred column order
    column_order = [
        "Model", "Type", "Architecture", "Optimizer", "Dropout", "LR",
        "Epoch (best)", "Train Acc", "Val Acc", "Val Loss", "Overfit Gap",
        "Avg Epoch Time (s)", "Converged by Epoch", "Stability (val acc)"
    ]
    df = df[column_order]
    df = df.sort_values("Val Acc", ascending=False).reset_index(drop=True)

    return df

def plot_model_comparison(df):
    '''
    Plot the model comparison.
    Args:
        df (pd.DataFrame): The dataframe to plot.
    '''
    df = df.sort_values("Val Acc", ascending=True)  # for horizontal view — better from smaller to larger

    plt.figure(figsize=(12, 8))
    bars = plt.barh(df["Model"], df["Val Acc"], color="skyblue")

    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{width:.3f}", va='center', fontsize=9)

    plt.xlabel("Validation Accuracy")
    plt.title("Model Comparison by Validation Accuracy")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_val_vs_test_acc(df):
    '''
    Plot the validation vs test accuracy.
    Args:
        df (pd.DataFrame): The dataframe to plot.
    '''
    df_sorted = df.sort_values("Val Acc", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df_sorted))

    ax.barh(y - 0.2, df_sorted["Val Acc"], height=0.4, label="Val Acc", alpha=0.7)    
    ax.barh(y + 0.2, df_sorted["Test Acc"], height=0.4, label="Test Acc", alpha=0.7)

    # Add values on bars
    for bar in ax.patches:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{width:.3f}", va='center', fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["Model"])
    ax.invert_yaxis()  # Highest accuracy at top

    ax.set_xlabel("Accuracy")
    ax.set_title("Validation vs Test Accuracy")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_dropout_vs_overfit(df):
    '''
    Plot the dropout vs overfit.
    Args:
        df (pd.DataFrame): The dataframe to plot.
    '''
    from adjustText import adjust_text

    df = df.sort_values("Val Acc", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df["Dropout Avg"], df["Overfit Gap"],
        s=100,
        c=df["Val Acc"],
        cmap="coolwarm",
        edgecolors="black"
    )

    texts = []
    for i, row in df.iterrows():
        texts.append(plt.text(row["Dropout Avg"] + 0.002, row["Overfit Gap"] + 0.01, row["Model"], fontsize=8))
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="->", color='gray', lw=0.5),
        expand_text=(1.5, 1.8),
        expand_points=(2.5, 2.5),
        force_text=(0.8, 0.8),
        force_points=(0.3, 0.3),
        lim=1500,
        only_move={'points': 'y', 'text': 'xy'}
    )

    ax.axhline(0, linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("Average Dropout")
    ax.set_ylabel("Overfitting Gap (Train Acc - Val Acc)")
    ax.set_title("Dropout Rate vs Overfitting Gap")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Validation Accuracy")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_efficiency(df):
    '''
    Plot the efficiency.
    Args:
        df (pd.DataFrame): The dataframe to plot.
    '''
    df_sorted = df.sort_values("Efficiency", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(df_sorted["Model"], df_sorted["Efficiency"], color="mediumseagreen")

    # Add value labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va='center', fontsize=9)

    ax.set_xlabel("Val Accuracy / Avg Epoch Time (s)")
    ax.set_title("Model Efficiency (Accuracy per Second)")
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_group_accuracy(df):
    '''
    Plot the group accuracy.
    Args:
        df (pd.DataFrame): The dataframe to plot.
    '''
    melted = df.melt(
        id_vars=["Model", "Type"],
        value_vars=["Val Acc", "Test Acc"],
        var_name="Metric",
        value_name="Accuracy"
    )

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=melted,
        y="Type", x="Accuracy", hue="Metric",
        palette="Set2", showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black"},
        width=0.5
    )

    plt.xlabel("Accuracy", fontsize=12)
    plt.ylabel("Architecture Type", fontsize=12)
    plt.title("Model Accuracy Distribution by Architecture Type", fontsize=14)
    plt.grid(True, axis='x', linestyle="--", linewidth=0.5)
    plt.legend(title="Metric", loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()

def interactive_model_comparison(test_results, class_names):
    '''
    Interactive model comparison.
    Args:
        test_results (dict): The test results.
        class_names (list): The class names.
    '''
    model_names = list(test_results.keys())

    dropdown1 = widgets.Dropdown(options=model_names, description="Model 1:")
    dropdown2 = widgets.Dropdown(options=model_names, description="Model 2:")
    compare_button = widgets.Button(description="Compare Models", button_style="info")

    def on_compare(change=None):
        clear_output(wait=True)
        compare_models(dropdown1.value, dropdown2.value, test_results, class_names)

    compare_button.on_click(on_compare)

    display(widgets.VBox([dropdown1, dropdown2, compare_button]))

def compare_models(model1_name, model2_name, test_results, class_names):
    '''
    Compare two models using confusion matrix and per-class F1-score.
    Args:
        model1_name (str): Name of the first model
        model2_name (str): Name of the second model
        test_results (dict): Dict with y_true and y_pred for each model
        class_names (list): List of class labels
    '''

    # F1-score comparison table
    def extract_f1_dataframe(y_true, y_pred):
        '''
        Extract the F1-score dataframe.
        Args:
            y_true (list): The true classes.
            y_pred (list): The predicted classes.
        Returns:
            pd.DataFrame: The F1-score dataframe.
        '''
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        return pd.DataFrame(report).T.loc[class_names]

    # Confusion matrices
    plot_confusion_matrix(
        test_results[model1_name]["y_pred"],
        test_results[model1_name]["y_true"],
        class_names=class_names,
        normalize=False,
        model_name=model1_name
    )

    plot_confusion_matrix(
        test_results[model2_name]["y_pred"],
        test_results[model2_name]["y_true"],
        class_names=class_names,
        normalize=False,
        model_name=model2_name
    )

    # Extract F1-scores
    df1 = extract_f1_dataframe(test_results[model1_name]["y_true"], test_results[model1_name]["y_pred"])
    df2 = extract_f1_dataframe(test_results[model2_name]["y_true"], test_results[model2_name]["y_pred"])

    # Combine and compare
    f1_comparison = pd.concat([df1["f1-score"], df2["f1-score"]], axis=1)
    f1_comparison.columns = [model1_name, model2_name]
    f1_comparison["Δ F1"] = (f1_comparison[model2_name] - f1_comparison[model1_name]).round(4)
    f1_comparison = f1_comparison.sort_values("Δ F1", ascending=False)

    print(f"\n📊 F1-score comparison by class:\n")
    display(f1_comparison)

    # Bar plot
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = f1_comparison["Δ F1"].apply(lambda x: "green" if x > 0 else "red")
    bars = ax.barh(f1_comparison.index, f1_comparison["Δ F1"], color=colors)

    ax.axvline(0, color="gray", linestyle="--")
    ax.set_title(f"Δ F1-score ({model2_name} − {model1_name})")
    ax.set_xlabel("Δ F1-score")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    # Add bar value annotations
    for bar in bars:
        value = bar.get_width()
        x = bar.get_x() + value
        ha = 'left' if value >= 0 else 'right'
        offset = 0.005 if value >= 0 else -0.005
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}", va='center', ha=ha, fontsize=9)

    fig.tight_layout()
    plt.show()
    plt.close()


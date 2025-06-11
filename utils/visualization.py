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
from collections import Counter, defaultdict

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import display, clear_output
import ipywidgets as widgets
from IPython.display import HTML

# PyTorch
import torch

# Model complexity
from ptflops import get_model_complexity_info

# Local
from utils.paths import MODELS_DIR

def display_df(df, rows = 10, show_index = False):
    """
    Display a DataFrame in HTML format.
    Args:
        df: DataFrame to display
        rows: Number of rows to display
        show_index: Whether to show row indices
    """
    if rows==0:
        display(HTML(df.to_html(index=show_index)))
    else:
        display(HTML(df.head(rows).to_html(index=show_index)))

# ------------------------------------------------------------------------------------------------
# Data visualization
# ------------------------------------------------------------------------------------------------

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

def print_per_class_accuracy(y_true, y_pred, class_names):
    '''
    Print the per-class accuracy.
    Args:
        y_true (list): The true classes.
        y_pred (list): The predicted classes.
        class_names (list): The class names.
    '''
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    for class_name, metrics in report.items():
        if class_name in class_names:
            print(f"{class_name}: {metrics['precision']:.4f} {metrics['recall']:.4f} {metrics['f1-score']:.4f}")

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
    Returns:
        pd.DataFrame: The comparison table.
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
        model_config = config.get("model_kwargs", {})
        model_type = config.get("model_type")

        # ------- Architecture summary -------
        if model_type == "CIFAR10_CNN":
            conv = model_config.get('conv_layers', "N/A")
            fc = model_config.get('fc_layers', "N/A")
            bn = model_config.get('batch_norm', False) or model_config.get('use_batchnorm', False)
            arch_parts = []
            if conv != "N/A":
                arch_parts.append(f"conv: {conv}")
            if fc != "N/A":
                arch_parts.append(f"fc: {fc}")
            if bn:
                arch_parts.append("batch_norm: True")
            layer_summary = ", ".join(arch_parts) if arch_parts else "N/A"
            dropout = model_config.get("dropout_rates", "N/A")
        elif model_type == "CIFAR10_FC":
            hidden = model_config.get('hidden_layers', "N/A")
            layer_summary = f"fc: {hidden}"
            dropout = model_config.get("dropout_rates", "N/A")
        elif model_type == "CIFAR10_DeepDropoutCNN":
            conv = model_config.get('conv_channels', "N/A")
            dropout_sched = model_config.get('dropout_schedule', "N/A")
            bn = model_config.get('batch_norm', False) or model_config.get('use_batchnorm', False)
            layer_summary = f"conv: {conv}"
            if bn:
                layer_summary += ", batch_norm: True"
            dropout = dropout_sched
        elif model_type in ["CIFAR10_DenseNet121", "CIFAR10_ResNet18"]:
            layer_summary = "Standard"
            dropout = "N/A"
        else:
            layer_summary = "N/A"
            dropout = "N/A"

        # ------- Optimizer summary -------
        opt_dict = config.get("optimizer", {})
        opt_name = opt_dict.get("name", "N/A")
        lr = opt_dict.get("kwargs", {}).get("lr", None)
        # sample: Adam (lr=0.001)
        opt_summary = f"{opt_name} (lr={lr})" if lr else opt_name

        # ------- Metrics -------
        overfitting_gap = round(best_epoch["train_accuracy"] - best_epoch["val_accuracy"], 4)
        val_stability = round(np.std([e["val_accuracy"] for e in last_epochs]), 6)
        avg_epoch_time = round(np.mean([e["epoch_time"] for e in metrics]), 2)
        threshold = 0.9 * best_epoch["val_accuracy"]
        epochs_to_threshold = next((e["epoch"] for e in metrics if e["val_accuracy"] >= threshold), None)

        rows.append({
            "Model": model_name,
            "Type": model_type,
            "Architecture": layer_summary,
            "Optimizer": opt_summary,
            "Dropout": str(dropout),
            "LR": round(lr, 6) if lr else None,
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

def plot_scatter_metrics(
    df, 
    x_col, 
    y_col, 
    color_col=None, 
    label_col=None, 
    cmap='coolwarm', 
    figsize=(12,6), 
    title=None,
    colorbar_label=None,
    label_fontsize=10
):
    """
    Universal 2D scatter plot for model analysis, with optional color mapping and labels.

    Args:
        df (pd.DataFrame): Table with results.
        x_col (str): Column for X axis.
        y_col (str): Column for Y axis.
        color_col (str, optional): Column for color encoding.
        label_col (str, optional): Column for text labels.
        cmap (str): Matplotlib colormap.
        figsize (tuple): Plot size.
        title (str): Title.
        colorbar_label (str, optional): Colorbar label.
        label_fontsize (int): Font size for point labels.

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    if color_col:
        sc = plt.scatter(df[x_col], df[y_col], c=df[color_col], cmap=cmap, s=80, alpha=0.85)
        cbar = plt.colorbar(sc)
        cbar.set_label(colorbar_label if colorbar_label else color_col)
    else:
        plt.scatter(df[x_col], df[y_col], s=80, alpha=0.85)
    if label_col:
        for i, row in df.iterrows():
            plt.text(row[x_col], row[y_col], str(row[label_col]), fontsize=label_fontsize, alpha=0.7)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or f"{y_col} vs {x_col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_boxplot_by_category(df, cat_col, metric_col="Test Acc", title=None):
    plt.figure(figsize=(10,6))
    sns.boxplot(x=cat_col, y=metric_col, data=df)
    plt.title(title or f"{metric_col} by {cat_col}")
    plt.grid(True, axis='y')
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

    count_dict = defaultdict(int)
    texts = []
    for i, row in df.iterrows():
        if pd.notna(row["Dropout Avg"]) and pd.notna(row["Overfit Gap"]):
            label = row["Model"]
            offset_x = 0.004 * count_dict[row["Dropout Avg"]]
            texts.append(
                ax.text(
                    row["Dropout Avg"] + offset_x,
                    row["Overfit Gap"] + 0.01,
                    label,
                    fontsize=9,
                    color="black",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
                )
            )
            count_dict[row["Dropout Avg"]] += 1

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="->", color='gray', lw=0.5, shrinkA=10, shrinkB=7),
        expand_text=(1.5, 1.8),
        expand_points=(2.0, 2.0),
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

# ------------------------------------------------------------------------------------------------
# Metrics analysis
# ------------------------------------------------------------------------------------------------
def extract_model_features(model):
    """
    Extracts structural features from a PyTorch model.
    Args:
        model (nn.Module): Initialized model.
    Returns:
        dict: Features and stats.
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1024**2

    num_layers = 0
    num_conv = 0
    num_fc = 0
    num_bn = 0
    num_relu = 0
    num_dropout = 0
    max_out_channels = 0
    layer_types = set()
    for m in model.modules():
        num_layers += 1
        t = type(m).__name__
        layer_types.add(t)
        if isinstance(m, torch.nn.Conv2d):
            num_conv += 1
            max_out_channels = max(max_out_channels, getattr(m, "out_channels", 0))
        elif isinstance(m, torch.nn.Linear):
            num_fc += 1
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            num_bn += 1
        elif isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU)):
            num_relu += 1
        elif isinstance(m, torch.nn.Dropout):
            num_dropout += 1
    has_dropout = num_dropout > 0
    features = {
        "n_params": n_params,
        "param_size_mb": round(param_size_mb, 3),
        "num_layers": num_layers,
        "num_conv_layers": num_conv,
        "num_fc_layers": num_fc,
        "num_bn_layers": num_bn,
        "num_relu_layers": num_relu,
        "num_dropout_layers": num_dropout,
        "max_out_channels": max_out_channels,
        "has_dropout": has_dropout,
        "layer_types": ",".join(sorted(layer_types))
    }
    try:
        input_shape = model.input_shape if hasattr(model, "input_shape") else (3, 32, 32)
        input_shape = tuple(int(x) for x in input_shape)
        flops, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)
        features["flops"] = flops
    except Exception as e:
        features["flops"] = None        

    return features

def build_extended_table(models_dir):
    """
    Build an extended comparison table with architectural and training features for analysis.
    Args:
        models_dir (str): Path to models folder (распакованный архив).
    Returns:
        pd.DataFrame
    """
    from core.cifar10_classifier import CIFAR10Classifier

    rows = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        config_file = [f for f in os.listdir(model_path) if f.endswith("_config.json")]
        metrics_file = [f for f in os.listdir(model_path) if f.endswith("_metrics.json")]
        test_file = [f for f in os.listdir(model_path) if f.endswith("_test.json")]
        if not config_file or not metrics_file:
            continue

        with open(os.path.join(model_path, config_file[0]), "r") as f:
            config = json.load(f)
        with open(os.path.join(model_path, metrics_file[0]), "r") as f:
            metrics = json.load(f)

        model_type = config.get("model_type")
        model_kwargs = config.get("model_kwargs", {})
        optimizer_cfg = config.get("optimizer", {})
        lr_sched_cfg = config.get("lr_scheduler", {})
        aug_cfg = config.get("augmentation", {})

        # Main architectural features
        if model_type == "CIFAR10_CNN":
            conv_layers = model_kwargs.get('conv_layers', [])
            fc_layers = model_kwargs.get('fc_layers', [])
            num_conv_layers = len(conv_layers)
            num_fc_layers = len(fc_layers)
            conv_channels_sum = sum(l.get('out_channels', 0) for l in conv_layers)
            batchnorm = any(l.get('batch_norm', False) for l in conv_layers)
            activation_fn = model_kwargs.get('activation_fn_name', "N/A")
            dropout = np.mean(model_kwargs.get('dropout_rates', [0]))
            input_size = 32  # standard if not specified otherwise
        elif model_type == "CIFAR10_FC":
            num_conv_layers = 0
            num_fc_layers = len(model_kwargs.get('hidden_layers', []))
            conv_channels_sum = 0
            batchnorm = False
            activation_fn = model_kwargs.get('activation_fn_name', "N/A")
            dropout = np.mean(model_kwargs.get('dropout_rates', [0]))
            input_size = 32
        elif model_type == "CIFAR10_DeepDropoutCNN":
            conv_channels = model_kwargs.get('conv_channels', [])
            num_conv_layers = len(conv_channels)
            num_fc_layers = 0  # not always explicitly present
            conv_channels_sum = sum(conv_channels)
            batchnorm = model_kwargs.get('use_batchnorm', False) or model_kwargs.get('batch_norm', False)
            activation_fn = model_kwargs.get('activation_fn_name', "N/A")
            dropout = np.mean(model_kwargs.get('dropout_schedule', [0]))
            input_size = model_kwargs.get('input_shape', [3, 32, 32])[-1]
        elif model_type in ["CIFAR10_DenseNet121", "CIFAR10_ResNet18"]:
            num_conv_layers = "standard"
            num_fc_layers = "standard"
            conv_channels_sum = "standard"
            batchnorm = True
            activation_fn = "ReLU"
            dropout = "N/A"
            input_size = 32
        else:
            num_conv_layers = num_fc_layers = conv_channels_sum = "N/A"
            batchnorm = activation_fn = dropout = input_size = "N/A"

        grayscale = config.get("grayscale", False) or model_kwargs.get("grayscale", False)

        # Training features
        optimizer = optimizer_cfg.get("name", "N/A")
        lr = optimizer_cfg.get("kwargs", {}).get("lr", None)
        lr_scheduler = lr_sched_cfg.get("name", "N/A")
        aug_mode = aug_cfg.get("mode", "off")
        mixup_alpha = aug_cfg.get("mixup_alpha", None)
        cutout_size = aug_cfg.get("cutout_size", None)

        # Metrics — take the best epoch by val_accuracy
        best_epoch = max(metrics, key=lambda e: e.get("val_accuracy", 0))
        last_epochs = metrics[-5:]
        overfit_gap = best_epoch["train_accuracy"] - best_epoch["val_accuracy"]
        val_stability = np.std([e["val_accuracy"] for e in last_epochs])
        avg_epoch_time = np.mean([e.get("epoch_time", 0) for e in metrics])
        threshold = 0.9 * best_epoch["val_accuracy"]
        epochs_to_threshold = next((e["epoch"] for e in metrics if e["val_accuracy"] >= threshold), None)

        try:
            # Build model from config (weights are not loaded)
            model = CIFAR10Classifier.build_from_config(config)
            model_features = extract_model_features(model)
        except Exception as e:
            print(f"Could not extract features for {model_name}: {e}")
            model_features = {
                "n_params": np.nan, "param_size_mb": np.nan,
                "num_layers": np.nan, "num_conv_layers": np.nan,
                "num_fc_layers": np.nan, "num_bn_layers": np.nan,
                "num_relu_layers": np.nan, "num_dropout_layers": np.nan,
                "max_out_channels": np.nan, "has_dropout": np.nan,
                "layer_types": np.nan
            }        

        row = {
            "Model": model_name,
            "Type": model_type,
            "num_conv_layers": num_conv_layers,
            "num_fc_layers": num_fc_layers,
            "conv_channels_sum": conv_channels_sum,
            "batchnorm": batchnorm,
            "activation_fn": activation_fn,
            "dropout": dropout,
            "input_size": input_size,
            "grayscale": grayscale,
            "optimizer": optimizer,
            "lr": lr,
            "lr_scheduler": lr_scheduler,
            "augmentation": aug_mode,
            "mixup_alpha": mixup_alpha,
            "cutout_size": cutout_size,
            "Epoch (best)": best_epoch["epoch"],
            "Train Acc": best_epoch["train_accuracy"],
            "Val Acc": best_epoch["val_accuracy"],
            "Val Loss": best_epoch["val_loss"],
            "Overfit Gap": overfit_gap,
            "Avg Epoch Time (s)": avg_epoch_time,
            "Converged by Epoch": epochs_to_threshold,
            "Stability (val acc)": val_stability,
            **model_features # add model features
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    numeric_cols = ['dropout', 'mixup_alpha', 'cutout_size', 'lr', 'conv_channels_sum', 'num_fc_layers']
    categorical_cols = ['activation_fn', 'optimizer', 'augmentation', 'lr_scheduler']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('N/A')    
    return df

def plot_correlation_heatmap(df, filter_type=None, metrics_cols=None, figsize=(10,8), annot=True, cmap="coolwarm", title="Correlation Heatmap of Model Metrics"):
    """
    Draws a correlation heatmap between numeric columns (metrics) of the model results dataframe.

    Args:
        df (pd.DataFrame): DataFrame with all model results and metrics.
        filter_type (str, optional): Filter by model type.
        metrics_cols (list, optional): List of columns to include in correlation analysis. If None, all numeric columns are used.
        figsize (tuple): Figure size for the plot.
        annot (bool): Annotate correlation coefficients.
        cmap (str): Colormap for the heatmap.
        title (str): Plot title.

    Returns:
        None
    """
    if metrics_cols is None:
        # By default, take all numeric columns
        metrics_cols = df.select_dtypes(include='number').columns.tolist()
    if filter_type:
        df = df[df["Type"] == filter_type]
    corr = df[metrics_cols].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap=cmap, fmt=".2f")
    plt.title(title + f" ({filter_type})")
    plt.tight_layout()
    plt.show()

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

def plot_feature_importance(df, filter_type=None, target_col='Test Acc', feature_cols=None, figsize=(10,6), title="Feature Importance for Test Accuracy"):
    """
    Computes and plots feature importances for predicting target (e.g., test accuracy) from experiment parameters.

    Args:
        df (pd.DataFrame): DataFrame with all experiment results.
        filter_type (str, optional): Filter by model type.
        target_col (str): Column to predict (e.g., 'Test Acc').
        feature_cols (list, optional): List of columns to use as features. If None, auto-detect.
        figsize (tuple): Figure size.
        title (str): Plot title.

    Returns:
        None
    """
    # Auto-detect features if not provided: all except target and metrics
    if feature_cols is None:
        exclude = [target_col, 'Val Acc', 'Val Loss', 'Test Loss', 'Train Acc', 'Overfit Gap', 'Stability (val acc)', 'Epoch (best)']
        feature_cols = [c for c in df.columns if c not in exclude]
    if filter_type:
        df = df[df["Type"] == filter_type]
    X = df[feature_cols].copy()
    # Encode categorical features
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    y = df[target_col]
    
    model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    
    # Make DataFrame for plotting
    fi = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    fi = fi.sort_values('Importance', ascending=True)
    
    plt.figure(figsize=figsize)
    plt.barh(fi['Feature'], fi['Importance'])
    plt.title(title + f" ({filter_type})")
    plt.xlabel("Importance (XGBoost Gain)")
    plt.tight_layout()
    plt.show()
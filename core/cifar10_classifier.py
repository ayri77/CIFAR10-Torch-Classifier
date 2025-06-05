"""
CIFAR-10 Classifier Implementation

This module provides a comprehensive classifier implementation for the CIFAR-10 dataset.
It includes functionality for model building, training, evaluation, and prediction.

The classifier supports both fully connected and convolutional neural network architectures,
with configurable training parameters, data augmentation, and model saving/loading capabilities.
"""

import sys
import os
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from utils.paths import MODELS_DIR, DATA_DIR, ARCHITECTURES_DIR, TENSORBOARD_DIR
from utils.data_utils import mixup_data, mixup_criterion
from utils.visualization import plot_training_history, plot_confusion_matrix

from core.cifar10_model import CIFAR10_FC, CIFAR10_CNN
import config

import operator 
from functools import reduce

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

import time
import json
from pathlib import Path
from PIL import Image
from pprint import pformat

import inspect

class CIFAR10Classifier:
    """
    A comprehensive classifier implementation for CIFAR-10 dataset.
    
    This class provides a complete pipeline for training, evaluating, and using
    neural network models on the CIFAR-10 dataset. It supports both FC and CNN
    architectures with configurable parameters and training options.
    
    Args:
        name (str): Unique identifier for the model
        input_shape (tuple): Input shape (channels, height, width)
        num_classes (int): Number of output classes
        activation_fn_name (str): Name of activation function to use
        model_class (nn.Module, optional): Custom model class to use
        model_kwargs (dict, optional): Additional arguments for model initialization
        device (torch.device, optional): Device to use for training
        optimizer_name (str): Name of optimizer to use
        optimizer_kwargs (dict): Arguments for optimizer initialization
        criterion_name (str): Name of loss function to use
        criterion_kwargs (dict): Arguments for criterion initialization
        lr_scheduler_name (str): Name of learning rate scheduler to use
        lr_scheduler_kwargs (dict): Arguments for learning rate scheduler
        mean (tuple): Mean values for input normalization
        std (tuple): Standard deviation values for input normalization
        augmentation (dict/str/bool): Data augmentation configuration
        grayscale (bool): Whether to convert inputs to grayscale
    """
    def __init__(
            self, 
            name: str, 
            input_shape=(3, 32, 32), 
            num_classes=config.NUM_CLASSES, 
            activation_fn_name = config.ACTIVATION_FN, 
            model_class=None,
            model_kwargs=None,            
            device=None,
            optimizer_name=config.OPTIMIZER,
            optimizer_kwargs=config.OPTIMIZER_KWARGS,
            criterion_name=config.CRITERION,
            criterion_kwargs=config.CRITERION_KWARGS,
            lr_scheduler_name=config.LR_SCHEDULER,
            lr_scheduler_kwargs=config.LR_SCHEDULER_KWARGS,
            mean=config.MEAN,
            std=config.STD,
            augmentation=config.AUGMENTATION,
            grayscale=config.GRAYSCALE,
        ):
        
        self.name = name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation_fn_name = activation_fn_name
        self.model_class = model_class or CIFAR10_FC
        self.model_kwargs = model_kwargs or {}

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_name = criterion_name
        self.criterion_kwargs = criterion_kwargs
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        self.grayscale = grayscale
        if isinstance(augmentation, dict):
            augmentation_mode = augmentation.get("mode", "basic")
            cutout_size = augmentation.get("cutout_size", 8)
        elif isinstance(augmentation, str):
            augmentation_mode = augmentation
        elif isinstance(augmentation, bool):
            augmentation_mode = "basic" if augmentation else None        
        if augmentation_mode in ["mixup", "both"]:
            self.mixup_enabled = True
            self.mixup_alpha = augmentation.get("mixup_alpha", 0.4)
        else:
            self.mixup_enabled = False
            self.mixup_alpha = None
        

    def build_model(self):
        """
        Builds the neural network model based on the specified configuration.
        
        This method initializes the model architecture with the configured parameters
        and moves it to the specified device (CPU/GPU).
        """
        # Base kwargs, shared
        base_kwargs = {
            "input_size": reduce(operator.mul, self.input_shape),
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "activation_fn": getattr(nn, self.activation_fn_name),
        }

        # Merge and filter kwargs based on model_class's signature
        all_kwargs = {**base_kwargs, **self.model_kwargs}
        model_signature = inspect.signature(self.model_class.__init__)
        valid_keys = model_signature.parameters.keys()
        filtered_kwargs = {
            k: v for k, v in all_kwargs.items() if k in valid_keys
        }

        self.model = self.model_class(**filtered_kwargs)
        self.model.to(self.device)     


    def compile(self):
        """
        Compiles the model by initializing the optimizer, loss function, and learning rate scheduler.
        
        This method sets up all the components needed for training the model.
        Raises ValueError if unsupported optimizer, criterion, or scheduler is specified.
        """
        if self.optimizer_name == "Adam":
            self.optimizer = Adam(self.model.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_name == "SGD":
            self.optimizer = SGD(self.model.parameters(), **self.optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        if self.criterion_name == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss(**self.criterion_kwargs)
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion_name}")
        
        if self.lr_scheduler_name == "StepLR":
            self.lr_scheduler = StepLR(self.optimizer, **self.lr_scheduler_kwargs)
        elif self.lr_scheduler_name == "ReduceLROnPlateau":
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, **self.lr_scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported lr scheduler: {self.lr_scheduler_name}")

    def train(
            self, 
            train_loader, 
            val_loader, 
            num_epochs: int = config.NUM_EPOCHS,
            early_stopping: bool = True,
            patience: int = config.PATIENCE,
            verbose: bool = True,
            log_tensorboard: bool = config.LOG_TENSORBOARD
        ):
        """
        Trains the model on the provided data loaders.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs (int): Number of training epochs
            early_stopping (bool): Whether to use early stopping
            patience (int): Number of epochs to wait for improvement before stopping
            verbose (bool): Whether to print training progress
            log_tensorboard (bool): Whether to log metrics to TensorBoard
            
        Returns:
            dict: Training history containing metrics for each epoch
        """
        # create the models directory if it doesn't exist
        models_dir = os.path.join(MODELS_DIR, self.name)        
        os.makedirs(models_dir, exist_ok=True)
        
        if log_tensorboard:           
            writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, self.name))


        print("\n" + "=" * 60)
        print("🚀 Training configuration:")
        print(f"🧱 Architecture:       {type(self.model).__name__}")
        print(f"📦 Model name:        {self.name}")
        print(f"📐 Input shape:       {self.input_shape}")

        if isinstance(self.model, CIFAR10_CNN):
            print("🔷 Conv layers:")
            print(pformat(self.model_kwargs.get("conv_layers"), indent=4, width=80))
            print(f"🔢 FC layers:          {self.model_kwargs.get('fc_layers')}")
            print(f"🎛 Dropout rates:      {self.model_kwargs.get('dropout_rates')}")
        elif isinstance(self.model, CIFAR10_FC):
            print(f"🔢 Hidden layers:      {self.model_kwargs.get('hidden_layers')}")
            print(f"🎛 Dropout rates:      {self.model_kwargs.get('dropout_rates')}")

        print(f"⚙️ Activation:        {self.activation_fn_name}")
        print(f"📈 Optimizer:         {self.optimizer_name}")
        print("   " + pformat(self.optimizer_kwargs, indent=4, width=80))

        print(f"🎯 Criterion:         {self.criterion_name}")
        print("   " + pformat(self.criterion_kwargs, indent=4, width=80))

        print(f"🎯 Lr scheduler:      {self.lr_scheduler_name}")
        print("   " + pformat(self.lr_scheduler_kwargs, indent=4, width=80))

        print(f"🧠 Device:            {self.device}")
        print(f"📊 Epochs:            {num_epochs}")
        print(f"🪄 Early stopping:    {early_stopping} (patience={patience})")
        print("=" * 60 + "\n")        

        overall_start_time = time.time()
        best_accuracy = 0
        best_val_loss = float('inf')
        patience_counter = 0 

        history = []
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                if self.mixup_enabled:
                    inputs, targets_a, targets_b, lam = mixup_data(
                        X_batch, y_batch, alpha=self.mixup_alpha, device=self.device
                    )
                    outputs = self.model(inputs)
                    loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                _, train_predicted = torch.max(outputs, 1)
                train_correct += (train_predicted == y_batch).sum().item()
                train_total += y_batch.size(0)

            train_accuracy = train_correct / train_total

            self.model.eval()
            val_loss = 0 
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.lr_scheduler:
                if self.lr_scheduler_name == "StepLR":
                    self.lr_scheduler.step()
                elif self.lr_scheduler_name == "ReduceLROnPlateau":
                    self.lr_scheduler.step(val_loss / len(val_loader))
                    # Log current learning rate
                    if verbose:
                        print(f"Epoch {epoch+1}: Learning Rate = {current_lr:.6f}")

            val_accuracy = correct / total
            epoch_time = time.time() - epoch_start_time

            history.append({
                "epoch": epoch + 1,
                "epoch_time": epoch_time,
                "train_loss": round(train_loss / len(train_loader), 4),
                "train_accuracy": round(train_accuracy, 4),
                "val_loss": round(val_loss / len(val_loader), 4),
                "val_accuracy": round(val_accuracy, 4),
                "learning_rate": current_lr
            })            

            if log_tensorboard:
                writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
                writer.add_scalar("Accuracy/train", train_accuracy, epoch)
                writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)
                writer.add_scalar("Accuracy/val", val_accuracy, epoch)
                writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)
                writer.add_scalar("LearningRate/current", current_lr, epoch)

            if verbose:
                print(f"[{epoch+1}/{num_epochs}] "
                    f"Train loss: {train_loss/len(train_loader):.4f}, acc: {train_accuracy:.4f} | "
                    f"Val loss: {val_loss/len(val_loader):.4f}, acc: {val_accuracy:.4f} | "
                    f"🕒 {epoch_time:.2f}s")

            # Early stopping logic
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
                        'learning_rate': current_lr
                    }, os.path.join(models_dir, f"{self.name}_best_model.pth"))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("⏹ Early stopping triggered.")
                        break
            else:
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_accuracy': val_accuracy,
                        'learning_rate': current_lr
                    }, os.path.join(models_dir, f"{self.name}_best_model.pth"))

        if log_tensorboard:
            writer.close()

        # save the history
        metrics_path = os.path.join(models_dir,  f"{self.name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=4)  
            print(f"Metrics saved to {metrics_path}")

        # save the config
        config_dict = {
            "model_name": self.name,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "activation": self.activation_fn_name,
            "optimizer": self.optimizer_name,
            "optimizer_kwargs": self.optimizer_kwargs,
            "criterion": self.criterion_name,
            "criterion_kwargs": self.criterion_kwargs,
            "lr_scheduler": self.lr_scheduler_name,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "device": str(self.device),
            "num_epochs": num_epochs,
            "early_stopping": early_stopping,
            "patience": patience,
            "mean": self.mean,
            "std": self.std,
            "augmentation": self.augmentation,
            "grayscale": self.grayscale
        }

        # Detect CNN-style model and store relevant architecture fields
        if isinstance(self.model, CIFAR10_CNN):
            config_dict["conv_layers"] = self.model_kwargs.get('conv_layers')
            config_dict["fc_layers"] = self.model_kwargs.get('fc_layers')
            config_dict["dropout_rates"] = self.model_kwargs.get('dropout_rates')
        elif isinstance(self.model, CIFAR10_FC):
            config_dict["hidden_layers"] = self.model_kwargs.get('hidden_layers')
            config_dict["dropout_rates"] = self.model_kwargs.get('dropout_rates')

        config_path = os.path.join(models_dir,  f"{self.name}_config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)              
            print(f"Config saved to {config_path}")

        total_time = time.time() - overall_start_time
        print("\n✅ Training finished!")
        print(f"🕒 Total training time: {total_time:.2f} seconds")
        if history:
            best_epoch = max(history, key=lambda e: e['val_accuracy'])
            print("\n📈 Best validation accuracy:")
            print(f"🏆 Epoch {best_epoch['epoch']} — acc: {best_epoch['val_accuracy']:.4f}, loss: {best_epoch['val_loss']:.4f}")
        print("="*60)        

    def evaluate(self, data_loader, verbose=True):
        """
        Evaluates the model on the provided data loader.
        
        Args:
            data_loader: DataLoader for evaluation data
            verbose (bool): Whether to print evaluation results
            
        Returns:
            dict: Dictionary containing evaluation results with keys:
                - ``loss`` (float): Average loss over the dataset
                - ``accuracy`` (float): Accuracy of the model
                - ``y_pred`` (list): Predicted class labels
                - ``y_true`` (list): Ground truth class labels
                - ``probs`` (list): Predicted class probabilities
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_probs = []
        all_labels = []        

        with torch.no_grad():
            correct = 0
            total = 0
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
                all_preds.extend(predicted.cpu().tolist())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().tolist())
                all_labels.extend(y_batch.cpu().tolist())                

        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)

        if verbose:
            print(f"Validation loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "y_pred": all_preds,
            "y_true": all_labels,
            "probs": all_probs
        }

    def summary(self):
        """
        Prints a summary of the model architecture and parameters.
        
        Uses torchinfo to display detailed information about the model structure,
        number of parameters, and memory usage.
        """
        if not hasattr(self, "model"):
            print("⚠️ Model is not built yet.")
            return

        input_size = (1, *self.input_shape)  # e.g., (1, 3, 32, 32)
        return summary(self.model, input_size=input_size, device=self.device)    
# --------------------------------------------------------
# Predictions
# --------------------------------------------------------
    def predict_image(self, path: str, transform, class_names: list, show_image: bool = True):
        """
        Predicts the class of a single image.
        
        Args:
            path (str): Path to the image file
            transform: Transform to apply to the image
            class_names (list): List of class names
            show_image (bool): Whether to display the image
            
        Returns:
            str: Predicted class name
        """
        # predict the image
        # path: path to the image
        # transform: transform to apply to the image
        # class_names: list of class names
        # return: predicted class name
        self.model.eval()
        with torch.no_grad():
            image = Image.open(path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)
            pred_class = class_names[predicted.item()]

            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            top3 = np.argsort(probs)[-3:][::-1]
            lines = [f"{class_names[i]}: {probs[i]:.2f}" for i in top3]            

            if show_image:
                plt.figure(figsize=(4, 4))
                plt.imshow(image)
                plt.title(f"Predicted: {pred_class}\n" + "\n".join(lines))
                plt.axis("off")
                plt.show()
            return pred_class

    def predict_images(self, directory: str, transform, class_names: list, show_images: bool = True, n_cols: int = 5):
        """
        Predicts classes for all images in a directory.
        
        Args:
            directory (str): Path to directory containing images
            transform: Transform to apply to images
            class_names (list): List of class names
            show_images (bool): Whether to display the images
            n_cols (int): Number of columns in the display grid
            
        Returns:
            list: List of predicted class names
        """
        # predict the images
        # directory: directory of the images
        # transform: transform to apply to the images
        # class_names: list of class names
        # return: list of predicted class names

        def get_image_paths_from_directory(directory: str, extensions={".jpg", ".jpeg", ".png"}) -> list:
            return [str(p) for p in Path(directory).rglob("*") if p.suffix.lower() in extensions]

        image_paths = get_image_paths_from_directory(directory)
        predictions = []

        if show_images:
            n_rows = (len(image_paths) + n_cols - 1) // n_cols
            plt.figure(figsize=(4 * n_cols, 4 * n_rows))

        for i, path in enumerate(image_paths):
            self.model.eval()
            with torch.no_grad():
                image = Image.open(path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                output = self.model(image_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                pred_idx = np.argmax(probs)
                pred_class = class_names[pred_idx]

                predictions.append((path, pred_class))

                if show_images:
                    top3 = np.argsort(probs)[-3:][::-1]
                    lines = [f"{class_names[i]}: {probs[i]:.2f}" for i in top3]

                    plt.subplot(n_rows, n_cols, i + 1)
                    plt.imshow(image)
                    plt.title(f"Predicted: {pred_class}\n" + "\n".join(lines), fontsize=10)
                    plt.axis("off")

        if show_images:
            plt.tight_layout()
            plt.show()

        return predictions
    
# --------------------------------------------------------
# Save and load
# --------------------------------------------------------

    def save(self, path):
        """
        Saves the model and its configuration to disk.
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Loads a saved model and its configuration from disk.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

    @classmethod
    def load_model(cls, model_name, config_path, model_path):
        """
        Loads a saved model and its configuration.
        
        Args:
            model_name (str): Name of the model
            config_path (str): Path to the model configuration
            model_path (str): Path to the saved model weights
            
        Returns:
            CIFAR10Classifier: Loaded classifier instance
        """
        cfg = cls.load_config(config_path)

        if "conv_layers" in cfg and "fc_layers" in cfg:
            model_class = CIFAR10_CNN
            model_kwargs = {
                "conv_layers": cfg["conv_layers"],
                "fc_layers": cfg["fc_layers"],
                "dropout_rates": cfg.get("dropout_rates", []),
                "activation_fn": getattr(nn, cfg.get("activation", "ReLU"))
            }
        else:
            model_class = CIFAR10_FC
            model_kwargs = {
                "hidden_layers": cfg["hidden_layers"],
                "dropout_rates": cfg.get("dropout_rates", []),
                "activation_fn": getattr(nn, cfg.get("activation", "ReLU"))
            }
        model = cls(
            name=model_name,
            model_class=model_class,
            model_kwargs=model_kwargs,
            input_shape=tuple(cfg["input_shape"]),
            num_classes=cfg["num_classes"],
            activation_fn_name=cfg.get("activation", "ReLU"),
            device=cfg.get("device", "cpu"),
            optimizer_name=cfg.get("optimizer", "Adam"),
            optimizer_kwargs=cfg.get("optimizer_kwargs", {}),
            criterion_name=cfg.get("criterion", "CrossEntropyLoss"),
            criterion_kwargs=cfg.get("criterion_kwargs", {}),
            mean=cfg.get("mean", config.MEAN),
            std=cfg.get("std", config.STD),
            augmentation=cfg.get("augmentation", config.AUGMENTATION),            
            grayscale=cfg.get("grayscale", config.GRAYSCALE),
        )
        model.build_model()
        model.compile()
        model.load(model_path)
        return model
    
    @staticmethod
    def load_config(config_path):
        """
        Loads model configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            dict: Model configuration
        """
        # load the config
        # config_path: path to load the config
        return json.load(open(config_path))
    
    @staticmethod
    def load_metrics(metrics_path):
        """
        Loads training metrics from a JSON file.
        
        Args:
            metrics_path (str): Path to the metrics file
            
        Returns:
            list: Training history metrics
        """
        # load the metrics
        # metrics_path: path to load the metrics
        return json.load(open(metrics_path))

# --------------------------------------------------------
# Plotting
# --------------------------------------------------------

    def plot_training_history(self, metrics_path):
        """
        Plots the training history using the loaded metrics.
        
        Args:
            metrics_path (str): Path to the metrics file
        """
        # load the metrics
        # metrics_path: path to load the metrics
        metrics = self.load_metrics(metrics_path)
        save_path = os.path.join(os.path.dirname(metrics_path), f"{self.name}_metrics.png")
        plot_training_history(metrics, save_path=save_path)

    def plot_confusion_matrix(self, y_pred_classes, y_true, class_names=None, normalize=False):
        """
        Plots the confusion matrix for model predictions.
        
        Args:
            y_pred_classes: Predicted class labels
            y_true: True class labels
            class_names (list, optional): List of class names
            normalize (bool): Whether to normalize the confusion matrix
        """
        # for compatibility with the utils.visualization.plot_confusion_matrix
        # plot the confusion matrix
        # y_pred_classes: predicted classes
        # y_true: true classes
        # class_names: list of class names
        # normalize: normalize the confusion matrix
        plot_confusion_matrix(y_pred_classes, y_true, class_names=class_names, normalize=normalize)

 
    def show_misclassified(self, data_loader, class_names=None, max_images=10):
        """
        Displays examples of misclassified images.
        
        Args:
            data_loader: DataLoader containing the images
            class_names (list, optional): List of class names
            max_images (int): Maximum number of misclassified images to show
        """
        self.model.eval()
        images = []
        predictions = []
        probs = []
        labels = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                _, predicted = torch.max(outputs, 1)

                for img, pred, true, prob in zip(X_batch.cpu(), predicted.cpu(), y_batch, batch_probs):
                    if pred != true:
                        images.append(img)
                        predictions.append(pred.item())
                        labels.append(true.item())
                        probs.append(prob)
                        if len(images) >= max_images:
                            break
                if len(images) >= max_images:
                    break

        # Plotting
        n_cols = (max_images + 1) // 2
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 6, 10))
        axes = axes.flatten()        
        
        for i in range(len(images)):
            # Augmented image
            img = images[i]
            if img.shape[0] == 1:
                img = img * torch.tensor(self.std) + torch.tensor(self.mean)
                img = img.squeeze(0)  # shape: [H, W]
            else:
                img = images[i].permute(1, 2, 0)
                img = img * torch.tensor(self.std) + torch.tensor(self.mean)  # denormalize
                img = torch.clamp(img, 0, 1)

            ax = axes[i]
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.axis("off")

            top3 = np.argsort(probs[i])[-3:][::-1]
            title = f"True: {class_names[labels[i]]}\nPred: {class_names[predictions[i]]}"
            subtitle = "\n".join([f"{class_names[j]}: {probs[i][j]:.2f}" for j in top3])
            ax.set_title(title + "\n" + subtitle, fontsize=12)

            # Add inset bar plot
            inset = ax.inset_axes([0.05, -0.35, 0.9, 0.3])
            inset.bar(range(len(probs[i])), probs[i], color="lightgray")
            inset.set_ylim(0, 1)
            inset.set_xticks(range(len(probs[i])))
            inset.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
            inset.set_yticks([0.0, 0.5, 1.0])
            inset.set_yticklabels(["0", "0.5", "1.0"], fontsize=9)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.subplots_adjust(wspace=0.5, hspace=2.5)
        plt.tight_layout()
        plt.show()

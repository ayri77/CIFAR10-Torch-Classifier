from cifar10_model import CIFAR10_torch
import config

import operator 
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

import os
import time
import json

from utils import show_random_samples, show_class_distribution, plot_training_history

class CIFAR10Classifier:
    def __init__(
            self, 
            name: str, 
            input_shape=(3, 32, 32), 
            num_classes=config.NUM_CLASSES, 
            hidden_layers=config.HIDDEN_LAYERS, 
            dropout_rates=config.DROPOUT_RATES, 
            activation_fn_name = config.ACTIVATION_FN, 
            device=None,
            optimizer_name=config.OPTIMIZER,
            optimizer_kwargs=config.OPTIMIZER_KWARGS,
            criterion_name=config.CRITERION,
            criterion_kwargs=config.CRITERION_KWARGS,
            mean=config.MEAN,
            std=config.STD
        ):
        
        self.name = name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rates = dropout_rates
        self.activation_fn_name = activation_fn_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_name = criterion_name
        self.criterion_kwargs = criterion_kwargs
        self.mean = mean
        self.std = std
    def build_model(self):
        input_size = reduce(operator.mul, self.input_shape)
        self.model = CIFAR10_torch(
            input_size=input_size,
            num_classes=self.num_classes,
            hidden_layers=self.hidden_layers,
            dropout_rates=self.dropout_rates,
            activation_fn=getattr(nn, self.activation_fn_name)
            )        
        self.model.to(self.device)          

    def compile(self):
        if self.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), **self.optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        if self.criterion_name == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss(**self.criterion_kwargs)
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion_name}")

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

        os.makedirs(f"models/{self.name}", exist_ok=True)

        if log_tensorboard:           
            writer = SummaryWriter()

        print("\n" + "="*60)
        print("🚀 Training configuration:")
        print(f"📦 Model name:        {self.name}")
        print(f"📐 Input shape:       {self.input_shape}")
        print(f"🔢 Hidden layers:     {self.hidden_layers}")
        print(f"🎛 Dropout rates:     {self.dropout_rates}")
        print(f"⚙️ Activation:        {self.activation_fn_name}")
        print(f"📈 Optimizer:         {self.optimizer_name} {self.optimizer_kwargs}")
        print(f"🎯 Criterion:         {self.criterion_name} {self.criterion_kwargs}")
        print(f"🧠 Device:            {self.device}")
        print(f"📊 Epochs:            {num_epochs}")
        print(f"🪄 Early stopping:    {early_stopping} (patience={patience})")
        print("="*60 + "\n")

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

            val_accuracy = correct / total
            epoch_time = time.time() - epoch_start_time

            history.append({
                "epoch": epoch + 1,
                "epoch_time": epoch_time,
                "train_loss": round(train_loss / len(train_loader), 4),
                "train_accuracy": round(train_accuracy, 4),
                "val_loss": round(val_loss / len(val_loader), 4),
                "val_accuracy": round(val_accuracy, 4),
            })            

            if log_tensorboard:
                writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
                writer.add_scalar("Accuracy/train", train_accuracy, epoch)
                writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)
                writer.add_scalar("Accuracy/val", val_accuracy, epoch)
                writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

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
                        'val_accuracy': val_accuracy
                    }, os.path.join("models", self.name, f"{self.name}_best_model.pth"))
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
                        'val_accuracy': val_accuracy
                    }, os.path.join("models", self.name, f"{self.name}_best_model.pth"))

        if log_tensorboard:
            writer.close()

        # save the history
        metrics_path = os.path.join("models", self.name,  f"{self.name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=4)  
            print(f"Metrics saved to {metrics_path}")

        # save the config
        config_dict = {
            "model_name": self.name,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "hidden_layers": self.hidden_layers,
            "dropout_rates": self.dropout_rates,
            "activation": self.activation_fn_name,
            "optimizer": self.optimizer_name,
            "optimizer_kwargs": self.optimizer_kwargs,
            "criterion": self.criterion_name,
            "criterion_kwargs": self.criterion_kwargs,
            "device": str(self.device),
            "num_epochs": num_epochs,
            "early_stopping": early_stopping,
            "patience": patience,
            "mean": self.mean,
            "std": self.std
        }

        config_path = os.path.join("models", self.name,  f"{self.name}_config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)              
            print(f"Config saved to {config_path}")

        total_time = time.time() - overall_start_time
        print("\n✅ Training finished!")
        print(f"🕒 Total training time: {total_time:.2f} seconds")
        print("="*60)        

    def evaluate(self, data_loader, verbose=True):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
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
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())                

        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)

        if verbose:
            print(f"Validation loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "y_pred": all_preds,
            "y_true": all_labels
        }
    
    def predict(self, data_loader):
        with torch.no_grad():
            outputs = []
            for batch in data_loader:
                X_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
                X_batch = X_batch.to(self.device)
                outputs.append(self.model(X_batch))

        return outputs

    def summary(self):        
        if not hasattr(self, "model"):
            print("⚠️ Model is not built yet.")
            return

        input_size = (1, *self.input_shape)  # e.g., (1, 3, 32, 32)
        return summary(self.model, input_size=input_size, device=self.device)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

    @classmethod
    def load_model(cls, model_name, config_path, model_path):
        cfg = cls.load_config(config_path)
        model = cls(
            name=model_name,
            input_shape=tuple(cfg["input_shape"]),
            num_classes=cfg["num_classes"],
            hidden_layers=cfg["hidden_layers"],
            dropout_rates=cfg["dropout_rates"],
            activation_fn_name=cfg["activation"],
            device=cfg.get("device", "cpu"),  # fallback to cpu
            optimizer_name=cfg.get("optimizer", "Adam"),
            optimizer_kwargs=cfg.get("optimizer_kwargs", {}),
            criterion_name=cfg.get("criterion", "CrossEntropyLoss"),
            criterion_kwargs=cfg.get("criterion_kwargs", {}),
            mean=cfg.get("mean", config.MEAN),
            std=cfg.get("std", config.STD)
        )
        model.build_model()
        model.compile()
        model.load(model_path)
        return model
    
    @staticmethod
    def load_config(config_path):
        # load the config
        # config_path: path to load the config
        return json.load(open(config_path))
    
    @staticmethod
    def load_metrics(metrics_path):
        # load the metrics
        # metrics_path: path to load the metrics
        return json.load(open(metrics_path))

    def plot_training_history(self, metrics_path):
        # load the metrics
        # metrics_path: path to load the metrics
        metrics = self.load_metrics(metrics_path)
        plot_training_history(metrics, save_path=os.path.join("models", self.name, f"{self.name}_metrics.png"))

    def plot_confusion_matrix(self, y_pred_classes, y_true, class_names=None, normalize=False):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred_classes, normalize='true' if normalize else None)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                    cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
   
    def show_misclassified(self, data_loader, class_names=None, max_images=10):
        self.model.eval()
        images = []
        predictions = []
        labels = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)

                for img, pred, true in zip(X_batch.cpu(), predicted.cpu(), y_batch):
                    if pred != true:
                        images.append(img)
                        predictions.append(pred.item())
                        labels.append(true.item())
                        if len(images) >= max_images:
                            break
                if len(images) >= max_images:
                    break

        # Plotting
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 6))
        for i in range(len(images)):
            plt.subplot(2, (max_images + 1) // 2, i + 1)
            img = images[i].permute(1, 2, 0)
            img = img * torch.tensor(self.std) + torch.tensor(self.mean)  # denormalize
            img = torch.clamp(img, 0, 1)
            plt.imshow(img)
            title = f"True: {class_names[labels[i]]}" if class_names else f"True: {labels[i]}"
            title += f"\nPred: {class_names[predictions[i]]}" if class_names else f", Pred: {predictions[i]}"
            plt.title(title, fontsize=10)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

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

class CIFAR10Classifier:
    def __init__(
            self, 
            name: str, 
            input_shape=(3, 32, 32), 
            num_classes=config.NUM_CLASSES, 
            hidden_layers=config.HIDDEN_LAYERS, 
            dropout_rate=config.DROPOUT_RATES, 
            activation_fn_name = config.ACTIVATION_FN, 
            device=None,
            optimizer_name=config.OPTIMIZER,
            optimizer_kwargs=config.OPTIMIZER_KWARGS,
            criterion_name=config.CRITERION,
            criterion_kwargs=config.CRITERION_KWARGS,
        ):
        
        self.name = name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation_fn_name = activation_fn_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_name = criterion_name
        self.criterion_kwargs = criterion_kwargs


    def build_model(self):
        input_size = reduce(operator.mul, self.input_shape)
        self.model = CIFAR10_torch(
            input_size=input_size,
            num_classes=self.num_classes,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
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
        print(f"🎛 Dropout rates:     {self.dropout_rate}")
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

        total_time = time.time() - overall_start_time
        print("\n✅ Training finished!")
        print(f"🕒 Total training time: {total_time:.2f} seconds")
        print("="*60)        

    def evaluate(self, data_loader, verbose=True):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

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

        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)

        if verbose:
            print(f"Validation loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")

        return {"loss": avg_loss, "accuracy": accuracy}

    
    def predict(self, data_loader):
        with torch.no_grad():
            outputs = []
            for batch in data_loader:
                X_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
                X_batch = X_batch.to(self.device)
                outputs.append(self.model(X_batch))

        return outputs

    def plot_confusion_matrix(self, y_pred_classes, y_true):
        # plot the confusion matrix
        # y_pred_classes: predicted classes
        # y_true: true classes

        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()    

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def load_best_model(self, path):
        self.load(f"models/{self.name}/best_model.pth")

    def summary(self):        
        if not hasattr(self, "model"):
            print("⚠️ Model is not built yet.")
            return

        input_size = (1, *self.input_shape)  # e.g., (1, 3, 32, 32)
        return summary(self.model, input_size=input_size, device=self.device)
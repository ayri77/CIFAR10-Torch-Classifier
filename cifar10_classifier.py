from cifar10_model import CIFAR10_torch
import torch
import torch.nn as nn
import operator 
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.optim as optim


class CIFAR10Classifier:
    def __init__(
            self, 
            name="fc_model", 
            input_shape=(3, 32, 32), 
            num_classes=10, 
            hidden_layers=[512, 256, 128], 
            dropout_rate=[0.3,0.3,0.3], 
            activation_fn = nn.ReLU, 
            lr=0.01, 
            device=None):
        
        self.name = name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.lr = lr
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        input_size = reduce(operator.mul, self.input_shape)
        self.model = CIFAR10_torch(
            input_size=input_size,
            num_classes=self.num_classes,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn
            )        
        self.model.to(self.device)          

    def compile(self,
            criterion=nn.CrossEntropyLoss, 
            optimizer=optim.SGD, 
            optimizer_kwargs={}, 
            criterion_kwargs={}):
        # loss function
        self.criterion = criterion(**criterion_kwargs)
        # optimizer
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)        

    def train(self, train_loader, val_loader, num_epochs=10, early_stopping=True, patience=30, verbose=True, log_tensorboard=False,):
        import os
        import time

        os.makedirs("models", exist_ok=True)
        if log_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()        
        
        start_time = time.time() 
        best_accuracy = 0
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0 

        for epoch in range(num_epochs):
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
            if log_tensorboard:
                writer.add_scalar("Accuracy/train", train_accuracy, epoch)
                writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)

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
                val_accuracy = correct/total

                if log_tensorboard:
                    writer.add_scalar("Loss/val", val_loss / len(val_loader), epoch)
                    writer.add_scalar("Accuracy/val", val_accuracy, epoch)    
                
                if verbose:
                    print(f"Epoch [{epoch+1}/{num_epochs}]. Train loss: {train_loss/len(train_loader):.4f}. "
                        f"Train acc: {train_accuracy:.4f}. "
                        f"Val loss: {val_loss/len(val_loader):.4f}. Val acc: {val_accuracy:.4f}")
            
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
                    }, os.path.join("models", "best_model.pth"))
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
                    }, os.path.join("models", "best_model.pth"))

                
        if log_tensorboard:
            writer.close()

        end_time = time.time()
        total_time = end_time - start_time

        print("\n" + "="*50)
        print(f"🏷️ Model: {self.name}")
        print(f"🔢 Hidden layers: {self.hidden_layers}")
        print(f"🎛️ Dropout: {self.dropout_rate}")
        print(f"⚙️ Activation: {self.activation_fn.__name__}")
        print(f"📦 Optimizer: {type(self.optimizer).__name__}")
        print(f"🧠 Device: {self.device}")
        print(f"📈 Epochs: {num_epochs}")
        print(f"🕒 Total training time: {total_time:.2f} seconds")
        print("="*50)
        

    def evaluate(self, data_loader):
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

        return {"loss": avg_loss, "accuracy": accuracy}
    
    def predict(self, data_loader):
        with torch.no_grad():
            outputs = []
            for X_batch in data_loader:
                X_batch = X_batch.to(self.device)
                outputs.append(self.model(X_batch))

        return outputs

    def plot_confusion_matrix(self, y_pred_classes, y_true):
        # plot the confusion matrix
        # y_pred_classes: predicted classes
        # y_true: true classes
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

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

    def summary(self):
        from torchinfo import summary
        if not hasattr(self, "model"):
            print("⚠️ Model is not built yet.")
            return

        input_size = (1, *self.input_shape)  # e.g., (1, 3, 32, 32)
        return summary(self.model, input_size=input_size, device=self.device)
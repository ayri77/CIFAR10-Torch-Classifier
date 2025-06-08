# CIFAR10 Torch Classifier 🧠🔥

A modular and extensible PyTorch project for training, evaluating, and comparing neural network architectures on the CIFAR-10 dataset.

## 📦 Project Status

✅ Current features:
- CIFAR-10 dataset loading and normalization
- Config-driven training pipeline
- Multiple model architectures (MLP, CNN)
- Advanced data augmentation (Mixup, Cutout)
- Reproducible training (seed, deterministic workers)
- TensorBoard logging
- Model saving with early stopping
- Comprehensive evaluation pipeline
- Model comparison utilities
- Kaggle competition submission support

## 📁 Project Structure

```
CIFAR10-Torch-Classifier/
│
├── architectures/          # Model architecture definitions
│   ├── mlp.py             # MLP model implementation
│   └── cnn.py             # CNN model implementation
│
├── core/                  # Core functionality
│   ├── cifar10_classifier.py  # Main classifier class
│   └── training.py        # Training utilities
│
├── utils/                 # Utility functions
│   ├── data_utils.py      # Data loading and preprocessing
│   ├── visualization.py   # Plotting and visualization
│   ├── utils.py           # General utilities
│   └── paths.py           # Path management
│
├── notebooks/             # Jupyter notebooks
│   ├── train.ipynb        # Training pipeline
│   ├── evaluate.ipynb     # Model evaluation
│   ├── compare_models.ipynb  # Model comparison
│   └── kaggle_competition_evaluation.ipynb  # Kaggle submission
│
├── models/                # Saved model checkpoints
├── runs/                  # TensorBoard logs
├── data/                  # Dataset storage
├── config.py              # Configuration settings
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## 📦 Features

### Model Architectures
- Fully-connected network (MLP) with configurable layers
- Convolutional Neural Network (CNN) with modern architecture
- ResNet18 and DenseNet121 variants adapted for 32×32 inputs
- Dropout, BatchNorm, Activation function selection
- Customizable layer configurations

### Training Features
- Advanced data augmentation (Mixup, Cutout)
- Learning rate scheduling
- Early stopping
- Model checkpointing
- TensorBoard logging (loss, accuracy, learning rate)
- Training history visualization

### Evaluation
- Comprehensive metrics calculation
- Confusion matrix visualization
- Per-class performance analysis
- Model comparison utilities
- Kaggle competition submission support

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/ayri77/CIFAR10-Torch-Classifier
cd CIFAR10-Torch-Classifier
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start training:
- Open `notebooks/train.ipynb` in Jupyter
- Follow the instructions in the notebook
- Monitor training progress in TensorBoard

## 📊 Model Performance

Current best results:
- MLP: ~75% validation accuracy
- CNN: ~80% validation accuracy

## 🧩 Author

Pavlo Borysov — [@ayri77](https://github.com/ayri77)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Project under active development. Feedback and contributions welcome!

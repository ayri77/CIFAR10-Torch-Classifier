# CIFAR10 Torch Classifier 🧠🔥

A modular and extensible PyTorch project for training, evaluating, and extending neural network architectures on the CIFAR-10 dataset.

## 📁 Project Structure

```
CIFAR10-Torch-Classifier/
│
├── cifar10_model.py        # PyTorch model definition (fully-connected)
├── cifar10_classifier.py   # Classifier class with build, compile, train, summary
├── train_fc.ipynb          # Training notebook for FC model
├── evaluate.ipynb          # [Planned] Evaluation and prediction analysis
├── utils.py                # [Planned] Utility functions
├── models/                 # Saved model checkpoints
├── runs/                   # TensorBoard logs
├── requirements.txt        # Required packages
└── README.md               # This file
```

## 📦 Features

- Fully-connected network with configurable architecture
- Dropout, BatchNorm, Activation function selection
- Training with validation and early stopping
- TensorBoard logging (loss, accuracy)
- Clear modular structure for extending to CNN and beyond

## ✅ TODO (See Issues)

- [x] Modularize model and training logic
- [x] Add training notebook
- [ ] Add evaluation notebook
- [ ] Add CNN support
- [ ] Export predictions for Kaggle
- [ ] Improve visualization and logging

## 🚀 Getting Started

```bash
git clone https://github.com/ayri77/CIFAR10-Torch-Classifier
cd CIFAR10-Torch-Classifier
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Launch the notebook and start training!

## 🧩 Author

Pavlo Borysov — [@ayri77](https://github.com/ayri77)

---

Project under active development. Feedback welcome!

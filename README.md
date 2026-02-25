# CIFAR10 Torch Classifier

Modular PyTorch framework for reproducible experimentation with image classification architectures on CIFAR-10.

The project focuses on structured experimentation, architecture comparison, and scalable training workflows rather than a single benchmark score.

---

## Problem

Design a flexible research-oriented training framework that allows:

- rapid experimentation with multiple CNN architectures
- configurable augmentation strategies
- reproducible training runs
- systematic model comparison
- structured evaluation and result tracking

---

## Architecture

The project is organized into clearly separated modules:

- **architectures/** – model definitions (MLP, CNN, ResNet18, DenseNet121 adapted for 32×32 inputs)
- **core/** – training loop abstraction and classifier logic
- **utils/** – data loading, augmentation, visualization, experiment utilities
- **notebooks/** – experiment workflows (training, evaluation, comparison, Kaggle submission)

Key design principles:

- config-driven training
- deterministic runs (seed control, worker control)
- modular augmentation pipeline (Mixup, Cutout)
- optimizer abstraction
- learning rate scheduling
- early stopping & checkpointing
- TensorBoard experiment tracking

---

## Experimentation Capabilities

- Multiple optimizers (SGD, Adam, AdamW, RMSprop, Adagrad)
- Configurable model hyperparameters
- Advanced augmentation strategies
- Validation monitoring
- Per-class evaluation and confusion matrix analysis
- Model comparison utilities
- Kaggle-compatible submission workflow

---

## Reproducibility

- Explicit configuration management
- Deterministic training setup
- Structured experiment logging
- Checkpoint-based model persistence

---

## Example Results

Baseline experiments achieved:

- MLP: ~75% validation accuracy
- CNN variants: ~80–84% validation accuracy

The framework is designed for extensibility (custom architectures, augmentation experiments, optimizer comparison).

---

## How to Run

```bash
git clone https://github.com/ayri77/CIFAR10-Torch-Classifier
cd CIFAR10-Torch-Classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

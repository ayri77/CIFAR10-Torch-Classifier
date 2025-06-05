"""
evaluate.py — Evaluate a trained CIFAR-10 model using saved config and weights.

Usage examples:
---------------
# Evaluate a saved model by name (folder inside /models)
python evaluate.py --model_name cnn_augmented

Arguments:
----------
--model_name     Folder name inside /models/ containing:
                 - {model_name}_config.json
                 - {model_name}_best_model.pth
                 - (optional) {model_name}_metrics.json for training plot

Notes:
------
- Evaluation is done on the CIFAR-10 test set.
- Applies the correct normalization and grayscale settings used during training.
- Outputs final test accuracy and loss.
- Also plots training history if metrics file is found.
"""

# Built-in
import os
import argparse
from IPython.display import display

# Third-party
import torch

# Environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Local
from utils.paths import MODELS_DIR
from config import BATCH_SIZE
from core.cifar10_classifier import CIFAR10Classifier
from utils.data_utils import (
    get_transforms,
    load_cifar10_datasets,
    create_loaders,
)

def main():
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(description="Evaluate saved CIFAR-10 model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate")
    args = parser.parse_args()    

    # Load model config and weights
    config_path = os.path.join(MODELS_DIR, args.model_name,  f"{args.model_name}_config.json")
    model_path = os.path.join(MODELS_DIR, args.model_name,  f"{args.model_name}_best_model.pth")    

    model = CIFAR10Classifier.load_model(
        model_name=args.model_name,
        config_path=config_path,
        model_path=model_path
    )

    display(model.summary())
    metrics_path = os.path.join(MODELS_DIR, args.model_name, f"{args.model_name}_metrics.json")
    display(model.plot_training_history(metrics_path))

    mean, std = torch.tensor(model.mean), torch.tensor(model.std)
    # Apply transformations
    full_transform = get_transforms(mean, std, augmentation=False, grayscale=model.grayscale)

    # Load with transformations
    _, test_dataset = load_cifar10_datasets(transform=full_transform, subset="test")

    # Loaders
    dummy_train_dataset, dummy_val_dataset = None, None
    _, _, test_loader = create_loaders(dummy_train_dataset, dummy_val_dataset, test_dataset, batch_size=BATCH_SIZE)    

    # Run evaluation
    metrics = model.evaluate(test_loader,verbose=False)
    print(f"\n📊 Test Accuracy: {metrics['accuracy']:.4f} | Test Loss: {metrics['loss']:.4f}")

if __name__ == "__main__":
    main()

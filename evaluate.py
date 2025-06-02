import argparse

import torch
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from config import BATCH_SIZE
from cifar10_classifier import CIFAR10Classifier
from utils import (
    get_transforms,
    load_cifar10_datasets,
    create_loaders,
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved CIFAR-10 model")
    parser.add_argument("--model_name", type=str, required=True, help="Path to folder with model and config")
    args = parser.parse_args()

    config_path = os.path.join("models", args.model_name,  f"{args.model_name}_config.json")
    model_path = os.path.join("models", args.model_name,  f"{args.model_name}_best_model.pth")    

    model = CIFAR10Classifier.load_model(
        model_name=args.model_name,
        config_path=config_path,
        model_path=model_path
    )

    model.summary()
    metrics_path = os.path.join("models", args.model_name, f"{args.model_name}_metrics.json")
    model.plot_training_history(metrics_path)

    mean, std = torch.tensor(model.mean), torch.tensor(model.std)
    # Apply transformations
    full_transform = get_transforms(mean, std)

    # Load with transformations
    _, test_dataset = load_cifar10_datasets(transform=full_transform, subset="test")

    # Loaders
    _, _, test_loader = create_loaders(_, _, test_dataset, batch_size=BATCH_SIZE)    

    # Run evaluation
    metrics = model.evaluate(test_loader,verbose=False)
    print(f"\n📊 Test Accuracy: {metrics['accuracy']:.4f} | Test Loss: {metrics['loss']:.4f}")

if __name__ == "__main__":
    main()

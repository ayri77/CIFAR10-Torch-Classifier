## Model Architecture Configuration Guide (`.json`)

This document summarizes all configurable options for CIFAR-10 model training using architecture `.json` files. Use it as a quick reference when defining or reviewing model configs.

---

### 🧠 Architecture Definition

| Section              | Setting       | Parameters                                                                                                 | Notes                                                                   |
| -------------------- | ------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `model_type`         | -             | `"CIFAR10_CNN"`, `"CIFAR10_FC"`, `"CIFAR10_ResNet18"`, `"CIFAR10_DenseNet121"`, `"CIFAR10_DeepDropoutCNN"` | Selects the model architecture type                                     |
| `conv_layers`        | layer configs | List of conv blocks: `out_channels`, `kernel_size`, `stride`, `padding`, `pool`, optional `batch_norm`     | Used for custom CNNs                                                    |
| `fc_layers`          | -             | List of integers, e.g. `[512, 256]`                                                                        | Defines sizes of fully connected layers                                 |
| `dropout_rates`      | -             | List of floats matching the length of `fc_layers` or `dropout_schedule` for Deep CNN                       | Defines dropout per layer (0.0 to 0.9). Use higher values to regularize |
| `activation_fn_name` | -             | One of: `"ReLU"`, `"LeakyReLU"`, `"GELU"`, `"Tanh"`, `"Sigmoid"`                                           | Activation function used throughout                                     |
| `model_kwargs`       | -             | Used for `CIFAR10_DeepDropoutCNN`, e.g. `conv_channels`, `dropout_schedule`, etc.                          | Passes internal parameters directly into advanced models                |

---

### ⚙️ Optimizer & Criterion

| Section     | Setting          | Parameters                                          | Notes                                    |
| ----------- | ---------------- | --------------------------------------------------- | ---------------------------------------- |
| `optimizer` | `name`, `kwargs` | E.g. `"Adam"`, `"SGD"`, etc. with `{ "lr": 0.001 }` | Learning algorithm used for model update |
| `criterion` | `name`, `kwargs` | E.g. `"CrossEntropyLoss"`                           | Loss function for classification         |

---

### 🔁 Learning Rate Scheduler

| Section        | Setting          | Parameters                                                                                                                        | Notes                                                                               |
| -------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `lr_scheduler` | `name`, `kwargs` | `"StepLR"` with `{ "step_size": 10, "gamma": 0.5 }`<br>`"ReduceLROnPlateau"` with `mode`, `factor`, `patience`, `threshold`, etc. | StepLR is static schedule; ReduceLROnPlateau is adaptive based on validation metric |

---

### 🎮 Input Transformations

| Section        | Setting                | Parameters                                                                            | Notes                                                               |
| -------------- | ---------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `augmentation` | `true/false` or object | If `true`, uses default transforms; or define as `{ mode, mixup_alpha, cutout_size }` | Controls data augmentation behavior (crop, flip, mixup, cutout)     |
| `grayscale`    | boolean                | `true` or `false`                                                                     | Converts RGB images to grayscale                                    |
| `Resize`       | list\[int]             | E.g. `[126, 126]`                                                                     | Resize input image (required for DeepDropoutCNN with large kernels) |

---

### 📂 Example: CIFAR10\_CNN

```json
{
  "model_type": "CIFAR10_CNN",
  "conv_layers": [
    { "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "pool": true },
    { "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "pool": true }
  ],
  "fc_layers": [512],
  "dropout_rates": [0.5],
  "activation_fn_name": "ReLU",
  "optimizer": {
    "name": "Adam",
    "kwargs": { "lr": 0.0005 }
  },
  "criterion": {
    "name": "CrossEntropyLoss",
    "kwargs": {}
  },
  "lr_scheduler": {
    "name": "ReduceLROnPlateau",
    "kwargs": {
      "mode": "min",
      "factor": 0.5,
      "patience": 3,
      "min_lr": 1e-6
    }
  },
  "augmentation": true,
  "grayscale": false
}
```

### 📂 Example: CIFAR10\_DeepDropoutCNN

```json
{
  "model_type": "CIFAR10_DeepDropoutCNN",
  "model_kwargs": {
    "input_shape": [3, 126, 126],
    "conv_channels": [320, 320, 640, 640, 960, 960, 1280, 1280, 1600, 1600, 1920, 1920],
    "dropout_schedule": [0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5],
    "activation_fn_name": "LeakyReLU"
  },
  "optimizer": {
    "name": "Adam",
    "kwargs": { "lr": 0.001 }
  },
  "criterion": {
    "name": "CrossEntropyLoss",
    "kwargs": {}
  },
  "lr_scheduler": {
    "name": "ReduceLROnPlateau",
    "kwargs": {
      "mode": "min",
      "factor": 0.5,
      "patience": 3,
      "threshold": 1e-5,
      "cooldown": 0,
      "min_lr": 1e-6,
      "verbose": true
    }
  },
  "augmentation": {
    "mode": "both",
    "mixup_alpha": 0.4,
    "cutout_size": 8
  },
  "grayscale": false,
  "Resize": [126, 126]
}
```

---

Feel free to extend this document as new settings are added to the project.

## Model Architecture Configuration Guide (`.json`)

This document summarizes all configurable options for CIFAR-10 model training using architecture `.json` files. Use it as a quick reference when defining or reviewing model configs.

---

### 🧠 Architecture Definition

| Section         | Setting       | Parameters                                                                              | Notes                                               |
| --------------- | ------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `model_type`    | -             | `"CIFAR10_CNN"` or `"CIFAR10_FC"`                                                       | Defines whether to use CNN or fully connected model |
| `conv_layers`   | layer configs | List of conv blocks with:<br>`out_channels`, `kernel_size`, `stride`, `padding`, `pool` | Used for CNN models                                 |
| `fc_layers`     | -             | List of integers, e.g. `[512, 256]`                                                     | Defines fully connected layers                      |
| `dropout_rates` | -             | List matching length of `fc_layers`                                                     | Set dropout per FC layer                            |
| `activation_fn` | -             | One of: `"ReLU"`, `"LeakyReLU"`, `"GELU"`, `"Tanh"`, `"Sigmoid"`                        | Activation function used throughout                 |

---

### ⚙️ Optimizer & Criterion

| Section     | Setting          | Parameters                                                         | Notes                            |
| ----------- | ---------------- | ------------------------------------------------------------------ | -------------------------------- |
| `optimizer` | `name`, `kwargs` | `"Adam"`, `"SGD"`, etc. <br> With `kwargs`: `lr`, `momentum`, etc. | Optimizer config                 |
| `criterion` | `name`, `kwargs` | Typically `"CrossEntropyLoss"`                                     | Loss function for classification |

---

### 🔁 Learning Rate Scheduler

| Section        | Setting          | Parameters                                                                                                                                | Notes                                                                                    |
| -------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `lr_scheduler` | `name`, `kwargs` | `"StepLR"` with `{ "step_size": 10, "gamma": 0.5 }`<br>`"ReduceLROnPlateau"` with:<br>`mode`, `factor`, `patience`, `min_lr`, `threshold` | StepLR: static schedule<br>ReduceLROnPlateau: adaptive, monitors `val_loss` or `val_acc` |

---

### 🎮 Input Transformations

| Section        | Setting | Parameters        | Notes                                                   |
| -------------- | ------- | ----------------- | ------------------------------------------------------- |
| `augmentation` | -       | `true` or `false` | Enables random crop, flip, color jitter, rotation, etc. |
| `grayscale`    | -       | `true` or `false` | Converts input images to single channel                 |

---

### 📂 Example JSON

```json
{
  "model_type": "CIFAR10_CNN",
  "conv_layers": [
    { "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "pool": true },
    { "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "pool": true }
  ],
  "fc_layers": [512],
  "dropout_rates": [0.5],
  "activation_fn": "ReLU",
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

---

Feel free to extend this document as new settings are added to the project.

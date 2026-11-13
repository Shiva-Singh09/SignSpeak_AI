import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 12

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_TO_IDX = {letter: i for i, letter in enumerate(LETTERS)}

DATA_DIR = Path("dataset/processed/realsign_features")
MODEL_DIR = Path("models")
REPORT_DIR = Path("outputs/reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "training_features.csv"
VAL_PATH = DATA_DIR / "validation_features.csv"

CHECKPOINT_PATH = MODEL_DIR / "signspeak_mlp_best.pth"
HISTORY_PATH = REPORT_DIR / "mlp_training_history.csv"


# --------------------------------------------------
# Reproducibility and device
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SIGNSPEAK AI - MLP TRAINING")
print("=" * 60)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Load feature CSV
# --------------------------------------------------

def load_dataset(path):
    features = []
    labels = []

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        feature_columns = reader.fieldnames[2:]

        if len(feature_columns) != 132:
            raise ValueError(
                f"Expected 132 features, found {len(feature_columns)}"
            )

        for row in reader:
            label = row["label"]

            if label not in CLASS_TO_IDX:
                raise ValueError(f"Unknown label: {label}")

            values = [
                float(row[column])
                for column in feature_columns
            ]

            features.append(values)
            labels.append(CLASS_TO_IDX[label])

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite features found in {path}")

    return x, y


print("\nLoading training data...")
x_train, y_train = load_dataset(TRAIN_PATH)

print("Loading validation data...")
x_val, y_val = load_dataset(VAL_PATH)

print("Training shape:", x_train.shape)
print("Validation shape:", x_val.shape)


# --------------------------------------------------
# Standardize coordinate features only.
# Fit statistics on training data only.
# The final 6 categorical/binary flags remain unchanged.
# --------------------------------------------------

COORDINATE_FEATURES = 126

mean = x_train[:, :COORDINATE_FEATURES].mean(
    axis=0, keepdims=True
)

std = x_train[:, :COORDINATE_FEATURES].std(
    axis=0, keepdims=True
)

std = np.maximum(std, 1e-6)

x_train[:, :COORDINATE_FEATURES] = (
    x_train[:, :COORDINATE_FEATURES] - mean
) / std

x_val[:, :COORDINATE_FEATURES] = (
    x_val[:, :COORDINATE_FEATURES] - mean
) / std

# Store the scaler for later webcam inference.
scaler_mean = mean.astype(np.float32).flatten()
scaler_std = std.astype(np.float32).flatten()


# --------------------------------------------------
# Convert to PyTorch tensors
# --------------------------------------------------

train_dataset = TensorDataset(
    torch.from_numpy(x_train),
    torch.from_numpy(y_train)
)

val_dataset = TensorDataset(
    torch.from_numpy(x_val),
    torch.from_numpy(y_val)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# --------------------------------------------------
# MLP architecture
# --------------------------------------------------

class SignLanguageMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(132, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),

            nn.Linear(128, 26)
        )

    def forward(self, x):
        return self.network(x)


model = SignLanguageMLP().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("\nModel architecture:")
print(model)

print(
    "\nTrainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)


# --------------------------------------------------
# Training and validation functions
# --------------------------------------------------

def run_training_epoch():
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def run_validation():
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += batch_size

    return total_loss / total, correct / total


# --------------------------------------------------
# Training loop with early stopping
# --------------------------------------------------

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):

    train_loss, train_accuracy = run_training_epoch()
    val_loss, val_accuracy = run_validation()

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
    })

    print(
        f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "class_to_idx": CLASS_TO_IDX,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
            "coordinate_features": COORDINATE_FEATURES,
            "input_features": 132,
            "architecture": "MLP_132_256_128_26",
        }, CHECKPOINT_PATH)

        print("  -> Best model checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch}. "
            f"Best epoch: {best_epoch}"
        )
        break


# --------------------------------------------------
# Save training history
# --------------------------------------------------

with HISTORY_PATH.open(
    "w", encoding="utf-8", newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
        ]
    )

    writer.writeheader()
    writer.writerows(history)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("Best epoch:", best_epoch)
print("Best validation loss:", round(best_val_loss, 4))

best_record = next(
    row for row in history if row["epoch"] == best_epoch
)

print(
    "Best validation accuracy:",
    f"{best_record['validation_accuracy'] * 100:.2f}%"
)

print("Checkpoint:", CHECKPOINT_PATH)
print("Training history:", HISTORY_PATH)

print("\nTesting data was not used during training.")

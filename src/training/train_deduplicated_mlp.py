
import csv
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path("dataset/processed/realsign_deduplicated")

FEATURE_DIR = BASE / "features"

MODEL_DIR = Path("models")
REPORT_DIR = Path("outputs/reports")
GRAPH_DIR = Path("outputs/graphs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = FEATURE_DIR / "training_features.csv"
VAL_FILE = FEATURE_DIR / "validation_features.csv"

CHECKPOINT_PATH = MODEL_DIR / "signspeak_mlp_deduplicated_v1.pth"
HISTORY_PATH = REPORT_DIR / "deduplicated_mlp_training_history.csv"

SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 12

INPUT_FEATURES = 132
COORDINATE_FEATURES = 126
NUM_CLASSES = 26


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA LOADING
# ============================================================

def load_csv(path, class_to_idx=None):
    features = []
    labels = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        feature_names = reader.fieldnames[2:]

        if len(feature_names) != INPUT_FEATURES:
            raise ValueError(f"Expected {INPUT_FEATURES} features, "f"found {len(feature_names)}"            )

        for row in reader:
            label = row["label"]

            if class_to_idx is None:
                labels.append(label)
            else:
                if label not in class_to_idx:
                    raise ValueError(f"Unknown label: {label}")

                labels.append(class_to_idx[label])

            features.append([
                float(row[name])
                for name in feature_names
            ])

    X = np.asarray(features, dtype=np.float32)

    if class_to_idx is None:
        return X, labels

    y = np.asarray(labels, dtype=np.int64)

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values found in {path}")

    return X, y


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class SignSpeakMLP(nn.Module):

    def __init__(self, input_features=132, num_classes=26):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            outputs = model(X)

            loss = criterion(outputs, y)

            total_loss += loss.item() * X.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# TRAINING
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("SIGNSPEAK - DEDUPLICATED DATASET MLP TRAINING V1")
    print("=" * 65)

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load training labels and create class mapping
    # --------------------------------------------------------

    X_train_raw, train_labels = load_csv(TRAIN_FILE)

    classes = sorted(set(train_labels))

    if len(classes) != NUM_CLASSES:
        raise ValueError(f"Expected 26 classes, found {len(classes)}")

    class_to_idx = {
        label: index
        for index, label in enumerate(classes)
    }

    y_train = np.asarray([
        class_to_idx[label]
        for label in train_labels
    ], dtype=np.int64)

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    X_val_raw, y_val = load_csv(VAL_FILE,class_to_idx)

    if X_train_raw.shape[1] != INPUT_FEATURES:
        raise ValueError("Training feature dimension mismatch")

    if X_val_raw.shape[1] != INPUT_FEATURES:
        raise ValueError("Validation feature dimension mismatch")

    print("\nDATASET")
    print("-" * 45)
    print(f"Training samples: {len(X_train_raw)}")
    print(f"Validation samples: {len(X_val_raw)}")
    print(f"Classes: {len(classes)}")
    print(f"Input features: {INPUT_FEATURES}")

    # --------------------------------------------------------
    # Standardization
    # Fit ONLY on training coordinates.
    # Presence and handedness flags remain unchanged.
    # --------------------------------------------------------

    scaler_mean = X_train_raw[:, :COORDINATE_FEATURES].mean(axis=0)

    scaler_std = X_train_raw[:, :COORDINATE_FEATURES].std(axis=0)

    scaler_std[scaler_std < 1e-8] = 1.0

    X_train = X_train_raw.copy()
    X_val = X_val_raw.copy()

    X_train[:, :COORDINATE_FEATURES] = (X_train[:, :COORDINATE_FEATURES] - scaler_mean) / scaler_std

    X_val[:, :COORDINATE_FEATURES] = (X_val[:, :COORDINATE_FEATURES] - scaler_mean) / scaler_std

    # --------------------------------------------------------
    # Convert to PyTorch tensors
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),torch.tensor(y_val, dtype=torch.long),)

    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=(device.type == "cuda"),)

    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=(device.type == "cuda"),)

    # --------------------------------------------------------
    # Initialize model
    # --------------------------------------------------------

    model = SignSpeakMLP(input_features=INPUT_FEATURES,num_classes=NUM_CLASSES,).to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nMODEL")
    print("-" * 45)
    print(model)
    print(f"Trainable parameters: {total_parameters:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    history = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    print("\nTRAINING")
    print("=" * 65)

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(X)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            train_loss_total += loss.item() * X.size(0)

            predictions = outputs.argmax(dim=1)

            train_correct += (predictions == y).sum().item()

            train_total += y.size(0)

        train_loss = train_loss_total / train_total
        train_accuracy = train_correct / train_total

        val_loss, val_accuracy = evaluate(model,val_loader,criterion,device,)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch [{epoch:03d}/{MAX_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        # Save checkpoint based on validation loss.
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(model.state_dict())

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(f"\nEarly stopping at epoch {epoch}.")

            break

    # --------------------------------------------------------
    # Restore best checkpoint
    # --------------------------------------------------------

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced")

    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "class_to_idx": class_to_idx,

        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,

        "input_features": INPUT_FEATURES,
        "coordinate_features": COORDINATE_FEATURES,
        "num_classes": NUM_CLASSES,

        "architecture": {
            "input": 132,
            "hidden_1": 256,
            "hidden_2": 128,
            "output": 26,
            "dropout_1": 0.30,
            "dropout_2": 0.20,
        },

        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,

        "training_config": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        },
    }

    torch.save(checkpoint, CHECKPOINT_PATH)

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    with HISTORY_PATH.open("w", encoding="utf-8", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    best_record = history[best_epoch - 1]

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Validation accuracy at best epoch: "f"{best_record['val_accuracy'] * 100:.2f}%")

    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"Training history: {HISTORY_PATH}")

    print("\nTesting split was not loaded or evaluated.")


if __name__ == "__main__":
    main()

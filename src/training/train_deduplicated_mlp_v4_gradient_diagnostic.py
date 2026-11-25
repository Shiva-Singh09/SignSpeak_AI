
from pathlib import Path
import copy
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (PROJECT_ROOT/ "dataset"/ "processed"/ "realsign_deduplicated"/ "features_hand_present")

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

# IMPORTANT:
# V4 uses separate output files.
# Existing V3 checkpoint and history remain untouched.

MODEL_PATH = (MODEL_DIR/ "signspeak_mlp_deduplicated_v4_gradient_diagnostic.pth")

HISTORY_PATH = (REPORT_DIR/ "deduplicated_v4_gradient_diagnostic_history.csv")

SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 100
PATIENCE = 12

EXPECTED_FEATURES = 132
COORDINATE_FEATURES = 126


# REPRODUCIBILITY

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# MODEL

class SignSpeakMLP(nn.Module):

    def __init__(self, input_size=132, num_classes=26):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        return self.network(x)


# GRADIENT DIAGNOSTICS

def collect_gradient_statistics(model):
    """
    Collect weight-gradient statistics after loss.backward()
    and before optimizer.step().

    Monitored layers:
        hidden_1: Linear(132, 256)
        hidden_2: Linear(256, 128)
        output:   Linear(128, 26)

    Metrics:
        grad_norm
        grad_mean
        grad_max
        zero_fraction

    NOTE:
    These metrics describe parameter weight gradients.
    They are diagnostic indicators, not standalone proof
    of vanishing gradients.
    """

    statistics = {}

    monitored_layers = {

        "hidden_1": model.network[0],

        "hidden_2": model.network[4],

        "output": model.network[8]
    }

    for layer_name, layer in monitored_layers.items():

        weight_grad = layer.weight.grad

        if weight_grad is None:

            statistics[f"{layer_name}_grad_available"] = 0

            statistics[f"{layer_name}_grad_norm"] = float("nan")

            statistics[f"{layer_name}_grad_mean"] = float("nan")

            statistics[f"{layer_name}_grad_max"] = float("nan")

            statistics[f"{layer_name}_zero_fraction"] = float("nan")

            continue

        grad = weight_grad.detach().float()

        statistics[f"{layer_name}_grad_available"] = 1

        statistics[f"{layer_name}_grad_norm"] = (torch.linalg.vector_norm(grad).item())

        statistics[f"{layer_name}_grad_mean"] = (grad.abs().mean().item())

        statistics[f"{layer_name}_grad_max"] = (grad.abs().max().item())

        statistics[f"{layer_name}_zero_fraction"] = ((grad == 0).float().mean().item())

    return statistics


# DATA LOADING

def load_split(split):

    path = DATA_DIR / f"{split}_features.csv"

    if not path.exists():

        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "label" not in df.columns:

        raise ValueError(f"Label column missing in {path}")

    feature_columns = [

        column
        for column in df.columns
        if column not in ("label", "image_name")
    ]

    if len(feature_columns) != EXPECTED_FEATURES:

        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"found {len(feature_columns)}"
        )

    X = df[feature_columns].to_numpy(dtype=np.float32)

    y_labels = df["label"].astype(str).to_numpy()

    if not np.isfinite(X).all():

        raise ValueError(f"Non-finite features found in {split}"        )

    return X, y_labels, feature_columns


# FEATURE STANDARDIZATION

def standardize_features(X, mean, std):

    X = X.copy()

    X[:, :COORDINATE_FEATURES] = (

        X[:, :COORDINATE_FEATURES] - mean) / std

    return X


# EVALUATION

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)

            labels = labels.to(device)

            logits = model(features)

            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    return total_loss / total, correct / total


# EPOCH GRADIENT SUMMARY

def summarize_epoch_gradients(batch_statistics):
    """
    Calculate sample-weighted epoch averages.

    Every batch contributes according to its number of samples.
    This prevents a smaller final batch from receiving the same
    weight as a full batch.
    """

    if not batch_statistics:

        raise RuntimeError("No gradient statistics collected during this epoch")

    summary = {}

    metric_names = [
        "grad_norm",
        "grad_mean",
        "grad_max",
        "zero_fraction"
    ]

    layer_names = [
        "hidden_1",
        "hidden_2",
        "output"
    ]

    for layer_name in layer_names:

        availability_key = (f"{layer_name}_grad_available")

        available_batches = [
            stats
            for stats in batch_statistics
            if stats[availability_key] == 1
        ]

        summary[availability_key] = (len(available_batches) / len(batch_statistics))

        for metric in metric_names:

            key = f"{layer_name}_{metric}"

            values = [
                stats[key]
                for stats in available_batches
            ]

            if values:
                summary[key] = float(np.mean(values))
            else:
                summary[key] = float("nan")

    return summary


# MAIN

def main():

    set_seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)

    print("SIGNSPEAK - HAND-PRESENT "
        "DEDUPLICATED MLP V4"
    )

    print("GRADIENT DIAGNOSTIC TRAINING")

    print("=" * 70)

    device = torch.device(

        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")


    # LOAD DATA

    print("\n[1/5] LOADING DATA")

    X_train, y_train_labels, feature_columns = (load_split("training"))

    X_val, y_val_labels, val_columns = (load_split("validation"))

    if feature_columns != val_columns:

        raise ValueError("Training and validation feature columns differ")

    classes = sorted(set(y_train_labels))

    if (
        len(classes) != 26
        or set(classes) != set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ):

        raise ValueError(
            "Expected all 26 alphabet classes"
        )

    if set(y_val_labels) != set(classes):

        raise ValueError(
            "Training and validation classes differ"
        )

    class_to_index = {

        label: index
        for index, label in enumerate(classes)
    }

    y_train = np.array(

        [
            class_to_index[label]
            for label in y_train_labels
        ],

        dtype=np.int64
    )

    y_val = np.array(

        [
            class_to_index[label]
            for label in y_val_labels
        ],

        dtype=np.int64
    )

    print(f"Training samples:   {len(X_train)}")

    print(f"Validation samples: {len(X_val)}")

    print(f"Features:           {X_train.shape[1]}")

    print(f"Classes:            {len(classes)}")

    # STANDARDIZATION

    print(
        "\n[2/5] STANDARDIZING COORDINATE FEATURES"
    )

    # Calculate statistics using training data only.

    mean = X_train[
        :, :COORDINATE_FEATURES
    ].mean(axis=0)

    std = X_train[
        :, :COORDINATE_FEATURES
    ].std(axis=0)

    std[std < 1e-8] = 1.0

    X_train = standardize_features(X_train,mean,std)

    X_val = standardize_features(X_val,mean,std)

    # DATA LOADERS

    train_dataset = TensorDataset(

        torch.tensor(
            X_train,
            dtype=torch.float32
        ),

        torch.tensor(
            y_train,
            dtype=torch.long
        )
    )

    val_dataset = TensorDataset(

        torch.tensor(
            X_val,
            dtype=torch.float32
        ),

        torch.tensor(
            y_val,
            dtype=torch.long
        )
    )

    generator = torch.Generator()

    generator.manual_seed(SEED)

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        generator=generator,

        num_workers=0
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )

    # MODEL SETUP
 

    print("\n[3/5] INITIALIZING MODEL")

    model = SignSpeakMLP(input_size=EXPECTED_FEATURES,num_classes=len(classes)).to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Parameters: {total_parameters:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )

    # TRAINING

    print("\n[4/5] TRAINING WITH GRADIENT MONITORING")

    best_val_loss = float("inf")

    best_val_accuracy = 0.0

    best_epoch = 0

    best_state = None

    patience_counter = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        train_loss_total = 0.0

        train_correct = 0

        train_total = 0

        # Reset diagnostics at the beginning of every epoch.

        epoch_gradient_stats = []

        for features, labels in train_loader:

            features = features.to(device)

            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(features)

            loss = criterion(logits, labels)

            # BACKPROPAGATION

            loss.backward()

            # GRADIENT DIAGNOSTICS
            # Read gradients before optimizer updates weights.

            gradient_stats = collect_gradient_statistics(model)

            epoch_gradient_stats.append(gradient_stats)

            # PARAMETER UPDATE

            optimizer.step()

            train_loss_total += (loss.item() * labels.size(0))

            predictions = logits.argmax(dim=1)

            train_correct += (predictions == labels).sum().item()

            train_total += labels.size(0)

        train_loss = (train_loss_total / train_total        )

        train_accuracy = (train_correct / train_total)

        # VALIDATION

        val_loss, val_accuracy = evaluate(model,val_loader,criterion,device)

        # GRADIENT SUMMARY

        gradient_summary = summarize_epoch_gradients(epoch_gradient_stats)

        # HISTORY

        epoch_record = {

            "epoch": epoch,

            "train_loss": train_loss,

            "train_accuracy": train_accuracy,

            "validation_loss": val_loss,

            "validation_accuracy": val_accuracy,

            **gradient_summary
        }

        history.append(epoch_record)

        # PRINT TRAINING METRICS

        print(

            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "

            f"Train Loss: {train_loss:.4f} | "

            f"Train Acc: {train_accuracy * 100:.2f}% | "

            f"Val Loss: {val_loss:.4f} | "

            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        # PRINT GRADIENT METRICS

        print(

            "  Gradient Norm | "

            f"H1: {gradient_summary['hidden_1_grad_norm']:.3e} | "

            f"H2: {gradient_summary['hidden_2_grad_norm']:.3e} | "

            f"OUT: {gradient_summary['output_grad_norm']:.3e}"
        )

        print(

            "  Gradient Mean | "

            f"H1: {gradient_summary['hidden_1_grad_mean']:.3e} | "

            f"H2: {gradient_summary['hidden_2_grad_mean']:.3e} | "

            f"OUT: {gradient_summary['output_grad_mean']:.3e}"
        )

        print(

            "  Zero Fraction | "

            f"H1: {gradient_summary['hidden_1_zero_fraction']:.4f} | "

            f"H2: {gradient_summary['hidden_2_zero_fraction']:.4f} | "

            f"OUT: {gradient_summary['output_zero_fraction']:.4f}"
        )

        # SAVE BEST MODEL IN MEMORY

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_val_accuracy = val_accuracy

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        # EARLY STOPPING

        if patience_counter >= PATIENCE:

            print(

                f"\nEarly stopping at epoch {epoch}. "

                f"Best epoch: {best_epoch}"
            )

            break

    # SAVE MODEL AND HISTORY

    print("\n[5/5] SAVING RESULTS")

    if best_state is None:

        raise RuntimeError(
            "Training did not produce a valid checkpoint"
        )

    checkpoint = {

        "model_state_dict": best_state,

        "input_size": EXPECTED_FEATURES,

        "num_classes": len(classes),

        "classes": classes,

        "feature_columns": feature_columns,

        "mean": mean,

        "std": std,

        "best_epoch": best_epoch,

        "best_val_loss": best_val_loss,

        "best_val_accuracy": best_val_accuracy,

        "architecture": "132-256-128-26",

        "experiment": "deduplicated_v4_gradient_diagnostic",

        "seed": SEED,

        "gradient_diagnostics": True,

        "gradient_metrics": [

            "grad_norm",

            "grad_mean",

            "grad_max",

            "zero_fraction"
        ],

        "monitored_layers": [

            "hidden_1",

            "hidden_2",

            "output"
        ],

        "optimizer": "AdamW",

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "batch_size": BATCH_SIZE
    }

    # V4-only checkpoint path.

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    history_df = pd.DataFrame(history)

    # V4-only training history path.

    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )

    # FINAL SUMMARY

    print("\n" + "=" * 70)

    print(
        "HAND-PRESENT V4 GRADIENT DIAGNOSTIC "
        "TRAINING COMPLETED"
    )

    print("=" * 70)

    print(f"Best epoch:           {best_epoch}")

    print(
        f"Best validation loss: {best_val_loss:.4f}"
    )

    print(
        f"Best validation acc:  "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"\nCheckpoint: "
        f"{MODEL_PATH.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"History:    "
        f"{HISTORY_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            f"\nTRAINING FAILED: {error}"
        )

        sys.exit(1)


import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


BASE = Path("dataset/processed/realsign_deduplicated")

FEATURE_FILE = BASE / "features" / "validation_features.csv"

CHECKPOINT_PATH = Path(
    "models/signspeak_mlp_deduplicated_v1.pth"
)

HISTORY_PATH = Path(
    "outputs/reports/deduplicated_mlp_training_history.csv"
)

REPORT_DIR = Path("outputs/reports")
GRAPH_DIR = Path("outputs/graphs")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


class SignSpeakMLP(nn.Module):

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

            nn.Linear(128, 26),
        )

    def forward(self, x):
        return self.network(x)


def load_history():

    with HISTORY_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:

        reader = csv.DictReader(file)

        history = list(reader)

    return {
        key: np.array(
            [float(row[key]) for row in history]
        )
        for key in [
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ]
    }


def plot_training_curves(history):

    epochs = history["epoch"]

    # Loss curve
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Deduplicated MLP V1 - Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_path = GRAPH_DIR / "deduplicated_v1_loss_curve.png"

    plt.savefig(loss_path, dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["train_accuracy"] * 100,
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"] * 100,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Deduplicated MLP V1 - Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    accuracy_path = (
        GRAPH_DIR / "deduplicated_v1_accuracy_curve.png"
    )

    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    best_loss_epoch = int(
        epochs[np.argmin(history["val_loss"])]
    )

    best_accuracy_epoch = int(
        epochs[np.argmax(history["val_accuracy"])]
    )

    print("\nTRAINING CURVE ANALYSIS")
    print("-" * 50)

    print(f"Best validation loss epoch: {best_loss_epoch}")

    print(
        f"Best validation accuracy epoch: "
        f"{best_accuracy_epoch}"
    )

    print(
        f"Highest validation accuracy: "
        f"{np.max(history['val_accuracy']) * 100:.2f}%"
    )

    print(f"Loss graph: {loss_path}")
    print(f"Accuracy graph: {accuracy_path}")


def load_validation_data(class_to_idx):

    X = []
    y = []
    image_names = []

    with FEATURE_FILE.open(
        "r", encoding="utf-8", newline=""
    ) as file:

        reader = csv.DictReader(file)

        feature_names = reader.fieldnames[2:]

        for row in reader:

            label = row["label"]

            X.append([
                float(row[name])
                for name in feature_names
            ])

            y.append(class_to_idx[label])
            image_names.append(row["image_name"])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        image_names,
    )


def main():

    print("=" * 60)
    print("DEDUPLICATED MLP V1 - VALIDATION ERROR ANALYSIS")
    print("=" * 60)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    class_to_idx = checkpoint["class_to_idx"]

    idx_to_class = {
        index: label
        for label, index in class_to_idx.items()
    }

    classes = [
        idx_to_class[index]
        for index in range(26)
    ]

    X, y_true, image_names = load_validation_data(
        class_to_idx
    )

    # Apply the saved training scaler.
    mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        checkpoint["scaler_std"],
        dtype=np.float32,
    )

    X[:, :126] = (
        X[:, :126] - mean
    ) / std

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SignSpeakMLP().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():

        logits = model(X_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = logits.argmax(dim=1)

        confidence = probabilities.max(dim=1).values

    y_pred = predictions.cpu().numpy()
    confidence = confidence.cpu().numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print(f"\nValidation samples: {len(y_true)}")
    print(f"Validation accuracy: {accuracy * 100:.2f}%")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(26)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    report_path = (
        REPORT_DIR / "deduplicated_v1_validation_report.csv"
    )

    with report_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "class",
            "precision",
            "recall",
            "f1_score",
            "support",
        ])

        for label in classes:

            values = report[label]

            writer.writerow([
                label,
                values["precision"],
                values["recall"],
                values["f1-score"],
                values["support"],
            ])

        for label in [
            "macro avg",
            "weighted avg",
        ]:

            values = report[label]

            writer.writerow([
                label,
                values["precision"],
                values["recall"],
                values["f1-score"],
                values["support"],
            ])

    print(f"\nClassification report: {report_path}")

    print("\nWORST PERFORMING CLASSES")
    print("-" * 75)

    sorted_classes = sorted(
        classes,
        key=lambda label: report[label]["f1-score"],
    )

    for label in sorted_classes[:10]:

        values = report[label]

        print(
            f"{label} | "
            f"Precision: {values['precision']:.3f} | "
            f"Recall: {values['recall']:.3f} | "
            f"F1: {values['f1-score']:.3f} | "
            f"Support: {int(values['support'])}"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(26)),
    )

    matrix_path = (
        REPORT_DIR / "deduplicated_v1_validation_confusion.csv"
    )

    with matrix_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["actual/predicted", *classes])

        for index, label in enumerate(classes):

            writer.writerow([
                label,
                *matrix[index].tolist(),
            ])

    plt.figure(figsize=(15, 13))

    plt.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto",
    )

    plt.title(
        "Deduplicated MLP V1 - Validation Confusion Matrix"
    )

    plt.xlabel("Predicted Alphabet")
    plt.ylabel("Actual Alphabet")

    plt.xticks(range(26), classes)
    plt.yticks(range(26), classes)

    plt.colorbar()

    plt.tight_layout()

    confusion_graph = (
        GRAPH_DIR / "deduplicated_v1_validation_confusion.png"
    )

    plt.savefig(
        confusion_graph,
        dpi=300,
    )

    plt.close()

    print(f"\nConfusion matrix CSV: {matrix_path}")
    print(f"Confusion matrix graph: {confusion_graph}")

    # --------------------------------------------------------
    # Most frequent incorrect predictions
    # --------------------------------------------------------

    errors = []

    for actual in range(26):

        for predicted in range(26):

            if actual == predicted:
                continue

            count = int(matrix[actual, predicted])

            if count > 0:

                errors.append((
                    count,
                    classes[actual],
                    classes[predicted],
                ))

    errors.sort(reverse=True)

    print("\nTOP 20 ALPHABET CONFUSIONS")
    print("-" * 55)

    for count, actual, predicted in errors[:20]:

        print(
            f"{actual} predicted as {predicted}: {count}"
        )

    # --------------------------------------------------------
    # Save individual prediction errors
    # --------------------------------------------------------

    error_path = (
        REPORT_DIR / "deduplicated_v1_validation_errors.csv"
    )

    with error_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image_name",
            "actual",
            "predicted",
            "confidence",
        ])

        for i in range(len(y_true)):

            if y_true[i] != y_pred[i]:

                writer.writerow([
                    image_names[i],
                    idx_to_class[int(y_true[i])],
                    idx_to_class[int(y_pred[i])],
                    float(confidence[i]),
                ])

    print(f"\nPrediction errors saved: {error_path}")

    # --------------------------------------------------------
    # Training curves
    # --------------------------------------------------------

    history = load_history()

    plot_training_curves(history)

    print("\n" + "=" * 60)
    print("VALIDATION ANALYSIS COMPLETED")
    print("Testing dataset was not accessed.")
    print("=" * 60)


if __name__ == "__main__":
    main()



from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "signspeak_mlp_best.pth"
TEST_DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "realsign_features"
    / "testing_features.csv"
)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
GRAPH_DIR = PROJECT_ROOT / "outputs" / "graphs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignLanguageMLP(nn.Module):
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


def calculate_metrics(y_true, y_pred, num_classes):
    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64
    )

    for actual, predicted in zip(y_true, y_pred):
        confusion[actual, predicted] += 1

    class_metrics = []

    for class_index in range(num_classes):
        tp = confusion[class_index, class_index]
        fp = confusion[:, class_index].sum() - tp
        fn = confusion[class_index, :].sum() - tp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        class_metrics.append({
            "class_index": class_index,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(confusion[class_index, :].sum())
        })

    accuracy = (
        np.trace(confusion) / confusion.sum()
        if confusion.sum() else 0.0
    )

    macro_precision = np.mean(
        [item["precision"] for item in class_metrics]
    )
    macro_recall = np.mean(
        [item["recall"] for item in class_metrics]
    )
    macro_f1 = np.mean(
        [item["f1_score"] for item in class_metrics]
    )

    return (
        confusion,
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        class_metrics
    )


def main():
    print("=" * 60)
    print("SIGNSPEAK AI - FINAL TEST SET EVALUATION")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(TEST_DATA_PATH)

    print(f"\nDevice: {DEVICE}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    input_size = checkpoint["input_features"]
    coordinate_features = checkpoint["coordinate_features"]

    scaler_mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32
    )

    scaler_std = np.asarray(
        checkpoint["scaler_std"],
        dtype=np.float32
    )

    input_features = [
        column for column in pd.read_csv(TEST_DATA_PATH, nrows=0).columns
        if column not in ("label", "image_name")
    ]

    if len(input_features) != input_size:
        raise ValueError(
            f"Expected {input_size} features, found {len(input_features)}"
        )
    scaler_std = np.asarray(
        checkpoint["scaler_std"],
        dtype=np.float32
    )

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {
        index: label
        for label, index in class_to_idx.items()
    }

    num_classes = len(class_to_idx)

    print(f"Number of classes: {num_classes}")
    print(f"Number of input features: {len(input_features)}")

    dataframe = pd.read_csv(TEST_DATA_PATH)

    if not all(feature in dataframe.columns for feature in input_features):
        raise ValueError("Test dataset is missing expected model features.")

    if "label" not in dataframe.columns:
        raise ValueError("Test dataset has no label column.")

    dataframe = dataframe[
        dataframe["label"].isin(class_to_idx)
    ].copy()

    X = dataframe[input_features].to_numpy(
    dtype=np.float32,
    copy=True)

    y = dataframe["label"].map(class_to_idx).to_numpy(
        dtype=np.int64
    )

    # Apply only the scaler statistics saved during training.
    # The first 126 columns are coordinate features.
    # The final 6 columns are binary presence and handedness flags.
    coordinate_count = int(coordinate_features)

    if coordinate_count != 126:
        raise ValueError(
            f"Expected 126 coordinate features, found {coordinate_count}"
        )

    if len(scaler_mean) != coordinate_count:
        raise ValueError("Scaler mean does not match coordinate feature count.")

    if len(scaler_std) != coordinate_count:
        raise ValueError("Scaler standard deviation does not match coordinate feature count.")

    X[:, :coordinate_count] = (
        X[:, :coordinate_count] - scaler_mean
    ) / scaler_std

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    test_loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=256,
        shuffle=False
    )

    model = SignLanguageMLP(
        input_size=len(input_features),
        num_classes=num_classes
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_predictions = []
    all_actual = []

    print("\nEvaluating test samples...")

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)

            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_actual.extend(labels.numpy())

    all_predictions = np.asarray(all_predictions)
    all_actual = np.asarray(all_actual)

    (
        confusion,
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        class_metrics
    ) = calculate_metrics(
        all_actual,
        all_predictions,
        num_classes
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Test samples:       {len(all_actual)}")
    print(f"Test accuracy:      {accuracy * 100:.2f}%")
    print(f"Macro precision:    {macro_precision * 100:.2f}%")
    print(f"Macro recall:       {macro_recall * 100:.2f}%")
    print(f"Macro F1-score:     {macro_f1 * 100:.2f}%")

    # Save overall metrics.
    metrics = {
        "test_samples": int(len(all_actual)),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1_score": float(macro_f1),
        "checkpoint": str(CHECKPOINT_PATH)
    }

    metrics_path = REPORT_DIR / "mlp_test_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Save per-class metrics.
    class_report = []

    for item in class_metrics:
        label = idx_to_class[item["class_index"]]

        class_report.append({
            "label": label,
            "precision": item["precision"],
            "recall": item["recall"],
            "f1_score": item["f1_score"],
            "support": item["support"]
        })

    class_report_path = REPORT_DIR / "mlp_classification_report.csv"

    pd.DataFrame(class_report).to_csv(
        class_report_path,
        index=False
    )

    # Save confusion matrix.
    labels = [
        idx_to_class[index]
        for index in range(num_classes)
    ]

    confusion_path = REPORT_DIR / "mlp_confusion_matrix.csv"

    pd.DataFrame(
        confusion,
        index=labels,
        columns=labels
    ).to_csv(confusion_path, index_label="Actual")

    # Plot confusion matrix.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 12))

    image = ax.imshow(confusion, interpolation="nearest")
    ax.set_title("SignSpeak AI - Test Confusion Matrix")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted Sign")
    ax.set_ylabel("Actual Sign")

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    fig.tight_layout()

    confusion_plot_path = GRAPH_DIR / "mlp_test_confusion_matrix.png"

    fig.savefig(confusion_plot_path, dpi=200)
    plt.close(fig)

    # Identify the most frequent misclassifications.
    mistakes = []

    for actual_index in range(num_classes):
        for predicted_index in range(num_classes):
            if actual_index != predicted_index:
                count = int(
                    confusion[actual_index, predicted_index]
                )

                if count > 0:
                    mistakes.append((
                        count,
                        idx_to_class[actual_index],
                        idx_to_class[predicted_index]
                    ))

    mistakes.sort(reverse=True)

    print("\nMost frequent misclassifications:")
    for count, actual, predicted in mistakes[:10]:
        print(f"Actual {actual} predicted as {predicted}: {count}")

    print("\nFiles saved:")
    print(metrics_path.relative_to(PROJECT_ROOT))
    print(class_report_path.relative_to(PROJECT_ROOT))
    print(confusion_path.relative_to(PROJECT_ROOT))
    print(confusion_plot_path.relative_to(PROJECT_ROOT))

    print("\nTest evaluation completed.")


if __name__ == "__main__":
    main()

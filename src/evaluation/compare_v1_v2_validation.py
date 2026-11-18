
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "dataset" / "processed"
    / "realsign_features" / "validation_features.csv"
)

MODEL_PATHS = {
    "V1": ROOT / "models" / "signspeak_mlp_best.pth",
    "V2": ROOT / "models" / "signspeak_mlp_v2.pth",
}

REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class SignLanguageMLP(nn.Module):

    def __init__(self, input_features):
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

            nn.Linear(128, 26)
        )

    def forward(self, x):
        return self.network(x)


def main():

    data = pd.read_csv(DATA_PATH)

    labels = data["label"].astype(str).to_numpy()

    results = {}

    print("=" * 65)
    print("V1 vs V2 - VALIDATION COMPARISON")
    print("=" * 65)
    print("Validation samples:", len(data))
    print("Device:", DEVICE)

    for version, checkpoint_path in MODEL_PATHS.items():

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False
        )

        input_features = int(checkpoint["input_features"])

        x = data.iloc[:, 2:2 + input_features].to_numpy(
            dtype=np.float32,
            copy=True
        )

        if x.shape[1] != input_features:
            raise ValueError(
                f"{version}: incorrect feature count"
            )

        coordinate_count = int(
            checkpoint["coordinate_features"]
        )

        mean = np.asarray(
            checkpoint["scaler_mean"],
            dtype=np.float32
        )

        std = np.asarray(
            checkpoint["scaler_std"],
            dtype=np.float32
        )

        if len(mean) != coordinate_count:
            raise ValueError(f"{version}: scaler mean mismatch")

        if len(std) != coordinate_count:
            raise ValueError(f"{version}: scaler std mismatch")

        x[:, :coordinate_count] = (
            x[:, :coordinate_count] - mean
        ) / std

        model = SignLanguageMLP(input_features).to(DEVICE)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        predictions = []

        with torch.no_grad():

            for start in range(0, len(x), 512):

                batch = torch.from_numpy(
                    x[start:start + 512]
                ).to(DEVICE)

                outputs = model(batch)

                predicted_indices = outputs.argmax(dim=1)

                predictions.extend(
                    predicted_indices.cpu().numpy().tolist()
                )

        idx_to_class = {
            int(index): label
            for label, index
            in checkpoint["class_to_idx"].items()
        }

        predicted_labels = [
            idx_to_class[int(index)]
            for index in predictions
        ]

        accuracy = accuracy_score(
            labels, predicted_labels
        )

        report = classification_report(
            labels,
            predicted_labels,
            labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            output_dict=True,
            zero_division=0
        )

        results[version] = {
            "accuracy": accuracy,
            "predictions": predicted_labels,
            "macro_f1": report["macro avg"]["f1-score"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
        }

        pd.DataFrame(report).transpose().to_csv(
            REPORT_DIR / f"{version.lower()}_validation_report.csv"
        )

        matrix = confusion_matrix(
            labels,
            predicted_labels,
            labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        )

        pd.DataFrame(
            matrix,
            index=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        ).to_csv(
            REPORT_DIR / f"{version.lower()}_validation_confusion.csv"
        )

        print(f"\n{version} RESULTS")
        print("-" * 40)
        print(f"Input features: {input_features}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Checkpoint validation loss: "
              f"{checkpoint['validation_loss']:.4f}")
        print(f"Validation accuracy: {accuracy * 100:.2f}%")
        print(f"Macro precision: "
              f"{results[version]['macro_precision'] * 100:.2f}%")
        print(f"Macro recall: "
              f"{results[version]['macro_recall'] * 100:.2f}%")
        print(f"Macro F1: "
              f"{results[version]['macro_f1'] * 100:.2f}%")

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    for metric in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1"
    ]:
        v1 = results["V1"][metric] * 100
        v2 = results["V2"][metric] * 100

        print(
            f"{metric:20s} | "
            f"V1: {v1:7.2f}% | "
            f"V2: {v2:7.2f}% | "
            f"Difference: {v2-v1:+.2f} pp"
        )

    changed = sum(
        a != b
        for a, b in zip(
            results["V1"]["predictions"],
            results["V2"]["predictions"]
        )
    )

    print("\nPredictions changed:", changed)
    print("Reports saved in:", REPORT_DIR)
    print("\nNo test data was used.")


if __name__ == "__main__":
    main()

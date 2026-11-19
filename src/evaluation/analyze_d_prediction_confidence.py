
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_FILE = Path(
    "dataset/processed/realsign_deduplicated/features/validation_features.csv"
)

CHECKPOINT_PATH = Path(
    "models/signspeak_mlp_deduplicated_v1.pth"
)

OUTPUT_DIR = Path(
    "outputs/reports/deduplicated_d_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "prediction_confidence_analysis.csv"


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


def main():

    print("=" * 65)
    print("DEDUPLICATED V1 - PREDICTION CONFIDENCE ANALYSIS")
    print("=" * 65)

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

    X = []
    labels = []
    image_names = []

    with FEATURE_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        feature_names = reader.fieldnames[2:]

        for row in reader:

            X.append([
                float(row[name])
                for name in feature_names
            ])

            labels.append(row["label"])
            image_names.append(row["image_name"])

    X = np.asarray(X, dtype=np.float32)

    mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        checkpoint["scaler_std"],
        dtype=np.float32,
    )

    X[:, :126] = (X[:, :126] - mean) / std

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SignSpeakMLP().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            torch.tensor(X, dtype=torch.float32).to(device)
        )

        probabilities = F.softmax(logits, dim=1)

        top2_values, top2_indices = torch.topk(
            probabilities,
            k=2,
            dim=1,
        )

    probabilities = probabilities.cpu().numpy()
    top2_values = top2_values.cpu().numpy()
    top2_indices = top2_indices.cpu().numpy()

    d_index = class_to_idx["D"]

    d_predictions = 0
    correct_d_predictions = 0
    incorrect_d_predictions = 0

    rows = []

    for i in range(len(labels)):

        actual = labels[i]

        predicted_index = int(top2_indices[i][0])

        predicted = idx_to_class[predicted_index]

        top1_probability = float(top2_values[i][0])

        top2_probability = float(top2_values[i][1])

        margin = top1_probability - top2_probability

        d_probability = float(
            probabilities[i][d_index]
        )

        correct = actual == predicted

        if predicted == "D":

            d_predictions += 1

            if correct:
                correct_d_predictions += 1
            else:
                incorrect_d_predictions += 1

        rows.append({

            "image_name": image_names[i],
            "actual": actual,
            "predicted": predicted,
            "correct": correct,
            "top1_probability": top1_probability,
            "top2_class": idx_to_class[int(top2_indices[i][1])],
            "top2_probability": top2_probability,
            "probability_margin": margin,
            "D_probability": d_probability,

        })

    rows.sort(
        key=lambda row: (
            row["predicted"] == "D",
            row["top1_probability"],
        ),
        reverse=True,
    )

    fieldnames = list(rows[0].keys())

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nPREDICTION SUMMARY")
    print("-" * 45)

    print(f"Validation samples: {len(labels)}")
    print(f"Predictions as D: {d_predictions}")
    print(f"Correct D predictions: {correct_d_predictions}")
    print(f"Incorrect D predictions: {incorrect_d_predictions}")

    print("\nINCORRECT D PREDICTIONS BY ACTUAL CLASS")
    print("-" * 45)

    for label in sorted(class_to_idx):

        matching = [
            row for row in rows
            if row["actual"] == label
            and row["predicted"] == "D"
            and not row["correct"]
        ]

        if matching:

            print(f"{label}: {len(matching)}")

    print("\nREPORT SAVED")
    print("-" * 45)
    print(OUTPUT_FILE)

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()

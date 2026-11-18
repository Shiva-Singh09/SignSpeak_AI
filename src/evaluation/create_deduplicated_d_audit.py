
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path("dataset/processed/realsign_deduplicated")

IMAGE_DIR = BASE / "images"

FEATURE_FILE = (
    BASE / "features" / "validation_features.csv"
)

CHECKPOINT_PATH = Path(
    "models/signspeak_mlp_deduplicated_v1.pth"
)

OUTPUT_DIR = Path(
    "outputs/reports/deduplicated_d_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

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


# ============================================================
# LOAD VALIDATION FEATURES
# ============================================================

def load_validation(checkpoint):

    X = []
    labels = []
    image_names = []

    class_to_idx = checkpoint["class_to_idx"]

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

            image_names.append(
                row["image_name"].strip()
            )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        checkpoint["scaler_std"],
        dtype=np.float32,
    )

    # Standardize coordinate features only.
    X[:, :126] = (X[:, :126] - mean) / std

    y = np.asarray(
        [
            class_to_idx[label]
            for label in labels
        ],
        dtype=np.int64,
    )

    return X, y, labels, image_names


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(
    title,
    samples,
    output_path,
):

    if not samples:

        print(
            f"No readable images available for {title}"
        )

        return

    columns = 4

    rows = (
        len(samples) + columns - 1
    ) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, rows * 3.2),
    )

    axes = np.asarray(axes).reshape(-1)

    for index, sample in enumerate(samples):

        image = sample["image"]

        actual = sample["actual"]

        confidence = sample["confidence"]

        image_name = sample["image_name"]

        axes[index].imshow(image)

        axes[index].set_title(
            f"Actual: {actual} | Predicted: D\n"
            f"Confidence: {confidence:.1%}\n"
            f"{image_name}",
            fontsize=8,
        )

        axes[index].axis("off")

    # Hide unused axes.
    for index in range(
        len(samples),
        len(axes),
    ):

        axes[index].axis("off")

    fig.suptitle(
        title,
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "DEDUPLICATED DATASET - D CLASS VISUAL AUDIT"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # 1. LOAD CHECKPOINT
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

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

    # --------------------------------------------------------
    # 2. LOAD VALIDATION DATA
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    X, y_true, labels, image_names = load_validation(
        checkpoint
    )

    print(
        f"\nValidation samples: {len(y_true)}"
    )

    # --------------------------------------------------------
    # 3. LOAD MODEL
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = SignSpeakMLP().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # 4. GENERATE PREDICTIONS
    # --------------------------------------------------------

    with torch.no_grad():

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
        ).to(device)

        logits = model(X_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = logits.argmax(
            dim=1
        )

        confidences = probabilities.max(
            dim=1
        ).values

    predictions = predictions.cpu().numpy()

    confidences = confidences.cpu().numpy()

    # --------------------------------------------------------
    # 5. CHECK D PREDICTIONS
    # --------------------------------------------------------

    d_index = class_to_idx["D"]

    total_d_predictions = int(
        np.sum(predictions == d_index)
    )

    actual_d_count = int(
        np.sum(y_true == d_index)
    )

    print("\nPREDICTION CHECK")
    print("-" * 45)

    print(
        f"Total predictions as D: {total_d_predictions}"
    )

    print(
        f"Actual D samples: {actual_d_count}"
    )

    # --------------------------------------------------------
    # 6. FIND IMAGES PREDICTED AS D
    # --------------------------------------------------------

    grouped = defaultdict(list)

    missing_paths = 0

    unreadable_images = 0

    found_images = 0

    printed_examples = 0

    for index in range(len(labels)):

        predicted_index = int(
            predictions[index]
        )

        predicted_label = idx_to_class[
            predicted_index
        ]

        if predicted_label != "D":

            continue

        actual_label = labels[index]

        image_name = image_names[index]

        image_path = IMAGE_DIR / Path(image_name)

        # Check whether the expected image exists.
        if not image_path.exists():

            missing_paths += 1

            if printed_examples < 5:

                print("\nMISSING IMAGE PATH")

                print(
                    "Actual label:",
                    actual_label,
                )

                print(
                    "CSV image_name:",
                    repr(image_name),
                )

                print(
                    "Expected path:",
                    image_path,
                )

                printed_examples += 1

            continue

        # Read image.
        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            unreadable_images += 1

            continue

        found_images += 1

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        grouped[actual_label].append({

            "image": image,

            "actual": actual_label,

            "predicted": "D",

            "confidence": float(
                confidences[index]
            ),

            "image_name": image_name,

            "image_path": str(image_path),

        })

    # --------------------------------------------------------
    # 7. IMAGE LOOKUP DIAGNOSTICS
    # --------------------------------------------------------

    print("\nIMAGE LOOKUP DIAGNOSTICS")
    print("-" * 45)

    print(
        f"Total predictions as D: {total_d_predictions}"
    )

    print(
        f"Images found: {found_images}"
    )

    print(
        f"Missing image paths: {missing_paths}"
    )

    print(
        f"Unreadable images: {unreadable_images}"
    )

    # --------------------------------------------------------
    # 8. GROUP-WISE PREDICTION COUNTS
    # --------------------------------------------------------

    print("\nPREDICTIONS AS D BY ACTUAL CLASS")
    print("-" * 45)

    for label in sorted(class_to_idx):

        count = len(
            grouped.get(label, [])
        )

        print(
            f"Actual {label} predicted as D: {count}"
        )

    # --------------------------------------------------------
    # 9. CREATE CONTACT SHEETS
    # --------------------------------------------------------

    selected_classes = [
        "D",
        "O",
        "Z",
        "V",
        "B",
        "U",
    ]

    print("\nCREATING CONTACT SHEETS")
    print("-" * 45)

    for label in selected_classes:

        samples = grouped.get(
            label,
            [],
        )

        if not samples:

            print(
                f"No readable examples for actual class {label}"
            )

            continue

        # Highest-confidence predictions first.
        samples = sorted(
            samples,
            key=lambda item: item["confidence"],
            reverse=True,
        )

        # Limit each sheet to 12 examples.
        samples = samples[:12]

        output_path = (
            OUTPUT_DIR
            / f"actual_{label}_predicted_D.png"
        )

        create_contact_sheet(
            f"Actual {label} - Predicted D",
            samples,
            output_path,
        )

    # --------------------------------------------------------
    # 10. COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "VISUAL AUDIT COMPLETED"
    )

    print(
        f"Output folder: {OUTPUT_DIR}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()

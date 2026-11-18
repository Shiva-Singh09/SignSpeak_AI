
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "dataset"
    / "processed"
    / "realsign_deduplicated"
    / "features_hand_present"
    / "validation_features.csv"
)

V1_PATH = ROOT / "models" / "signspeak_mlp_deduplicated_v1.pth"

V2_PATH = (
    ROOT
    / "models"
    / "signspeak_mlp_deduplicated_v2_hand_present.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "reports"
    / "model_comparison_hand_present"
)

COORDINATE_FEATURES = 126
BATCH_SIZE = 512


# ============================================================
# EXACT MODEL ARCHITECTURES
# ============================================================

class V1Model(nn.Module):
    """
    V1:
    Linear -> ReLU -> BatchNorm -> Dropout
    Linear -> ReLU -> BatchNorm -> Dropout
    Linear
    """

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


class V2Model(nn.Module):
    """
    V2:
    Linear -> BatchNorm -> ReLU -> Dropout
    Linear -> BatchNorm -> ReLU -> Dropout
    Linear
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(132, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 26)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(path):

    if not path.exists():
        raise FileNotFoundError(path)

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


def prepare_model(checkpoint, version, device):

    if version == "V1":

        model = V1Model()

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        class_to_idx = checkpoint["class_to_idx"]

        mean = np.asarray(
            checkpoint["scaler_mean"],
            dtype=np.float32
        )

        std = np.asarray(
            checkpoint["scaler_std"],
            dtype=np.float32
        )

    else:

        model = V2Model()

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        classes = checkpoint["classes"]

        class_to_idx = {
            label: index
            for index, label in enumerate(classes)
        }

        mean = np.asarray(
            checkpoint["mean"],
            dtype=np.float32
        )

        std = np.asarray(
            checkpoint["std"],
            dtype=np.float32
        )

    if len(class_to_idx) != 26:
        raise ValueError(f"{version}: Invalid class mapping")

    if len(mean) != COORDINATE_FEATURES:
        raise ValueError(f"{version}: Invalid normalization mean")

    if len(std) != COORDINATE_FEATURES:
        raise ValueError(f"{version}: Invalid normalization std")

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError(f"{version}: Non-finite normalization values")

    if np.any(std <= 0):
        raise ValueError(f"{version}: Invalid standard deviation")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_to_idx, mean, std


# ============================================================
# PREDICTION
# ============================================================

def predict(model, X, mean, std, device):

    X = X.copy()

    # Normalize coordinate features only.
    X[:, :COORDINATE_FEATURES] = (
        X[:, :COORDINATE_FEATURES] - mean
    ) / std

    predictions = []
    probabilities = []

    with torch.inference_mode():

        for start in range(0, len(X), BATCH_SIZE):

            batch = torch.tensor(
                X[start:start + BATCH_SIZE],
                dtype=torch.float32,
                device=device
            )

            logits = model(batch)

            probs = torch.softmax(logits, dim=1)

            predictions.extend(
                probs.argmax(dim=1).cpu().numpy()
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

    return (
        np.asarray(predictions, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float32)
    )


# ============================================================
# METRICS AND REPORTS
# ============================================================

def evaluate_model(
    name,
    model,
    X,
    y_true,
    classes,
    mean,
    std,
    device,
    image_names
):

    y_pred, probabilities = predict(
        model,
        X,
        mean,
        std,
        device
    )

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(26)),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(26))
    )

    pd.DataFrame(
        report
    ).transpose().to_csv(
        OUTPUT_DIR / f"{name.lower()}_class_report.csv"
    )

    pd.DataFrame(
        confusion,
        index=classes,
        columns=classes
    ).to_csv(
        OUTPUT_DIR / f"{name.lower()}_confusion_matrix.csv"
    )

    prediction_df = pd.DataFrame({
        "image_name": image_names,
        "actual": [classes[i] for i in y_true],
        "predicted": [classes[i] for i in y_pred],
        "correct": y_true == y_pred,
        "top1_confidence": probabilities.max(axis=1)
    })

    prediction_df.to_csv(
        OUTPUT_DIR / f"{name.lower()}_predictions.csv",
        index=False
    )

    print(f"\n{name} RESULTS")
    print("-" * 55)
    print(f"Accuracy:         {accuracy * 100:.4f}%")
    print(f"Macro precision:  {precision * 100:.4f}%")
    print(f"Macro recall:     {recall * 100:.4f}%")
    print(f"Macro F1:         {f1 * 100:.4f}%")

    print("\nWorst 5 classes by F1:")

    class_scores = []

    for letter in classes:
        class_scores.append(
            (letter, report[letter]["f1-score"])
        )

    class_scores.sort(key=lambda item: item[1])

    for letter, score in class_scores[:5]:
        print(f"{letter}: {score * 100:.2f}%")

    print("\nMisclassified samples:", int((y_true != y_pred).sum()))

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "correct": int((y_true == y_pred).sum()),
        "incorrect": int((y_true != y_pred).sum()),
        "samples": int(len(y_true))
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SIGNSPEAK - V1 VS V2 HAND-PRESENT VALIDATION COMPARISON")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    print("\n[1/4] LOADING COMMON VALIDATION DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    feature_columns = [
        col for col in df.columns
        if col not in ("label", "image_name")
    ]

    if len(feature_columns) != 132:
        raise ValueError(
            f"Expected 132 features, got {len(feature_columns)}"
        )

    if "image_name" not in df.columns:
        raise ValueError("image_name column missing")

    if df["label"].isna().any():
        raise ValueError("Missing validation labels")

    X = df[feature_columns].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite validation features")

    hand_0 = pd.to_numeric(df["hand_0_present"])
    hand_1 = pd.to_numeric(df["hand_1_present"])

    no_hand = ((hand_0 == 0) & (hand_1 == 0)).sum()

    if no_hand != 0:
        raise ValueError(
            f"Expected hand-present validation data, found {no_hand} no-hand rows"
        )

    image_names = df["image_name"].astype(str).to_numpy()

    # --------------------------------------------------------
    # LOAD CHECKPOINTS
    # --------------------------------------------------------

    print("\n[2/4] LOADING CHECKPOINTS")

    v1_checkpoint = load_checkpoint(V1_PATH)
    v2_checkpoint = load_checkpoint(V2_PATH)

    v1, v1_mapping, v1_mean, v1_std = prepare_model(
        v1_checkpoint, "V1", device
    )

    v2, v2_mapping, v2_mean, v2_std = prepare_model(
        v2_checkpoint, "V2", device
    )

    classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    expected_mapping = {
        letter: index
        for index, letter in enumerate(classes)
    }

    if v1_mapping != expected_mapping:
        raise ValueError("V1 class mapping does not match alphabet order")

    if v2_mapping != expected_mapping:
        raise ValueError("V2 class mapping does not match alphabet order")

    if "feature_columns" in v2_checkpoint:
        if feature_columns != v2_checkpoint["feature_columns"]:
            raise ValueError("V2 feature ordering does not match validation data")

    y_true = np.array(
        [expected_mapping[str(label)] for label in df["label"]],
        dtype=np.int64
    )

    print("Validation samples:", len(df))
    print("Features:", len(feature_columns))
    print("Classes:", len(classes))
    print("No-hand rows:", no_hand)
    print("Class mapping: verified")

    # --------------------------------------------------------
    # EVALUATE BOTH MODELS
    # --------------------------------------------------------

    print("\n[3/4] EVALUATING BOTH MODELS")

    v1_results = evaluate_model(
        "V1",
        v1,
        X,
        y_true,
        classes,
        v1_mean,
        v1_std,
        device,
        image_names
    )

    v2_results = evaluate_model(
        "V2",
        v2,
        X,
        y_true,
        classes,
        v2_mean,
        v2_std,
        device,
        image_names
    )

    # --------------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------------

    print("\n[4/4] SAVING COMPARISON")

    comparison = {
        "validation_samples": len(df),
        "validation_dataset": "hand_present",
        "v1": v1_results,
        "v2": v2_results,
        "accuracy_difference_percentage_points": (
            v2_results["accuracy"] - v1_results["accuracy"]
        ) * 100,
        "macro_f1_difference_percentage_points": (
            v2_results["macro_f1"] - v1_results["macro_f1"]
        ) * 100,
        "architecture_note": (
            "V1 uses Linear-ReLU-BatchNorm-Dropout; "
            "V2 uses Linear-BatchNorm-ReLU-Dropout. "
            "Therefore this is a model comparison, not an isolated "
            "no-hand filtering experiment."
        )
    }

    with open(
        OUTPUT_DIR / "comparison_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(comparison, file, indent=4)

    comparison_table = pd.DataFrame([
        {"Model": "V1", **v1_results},
        {"Model": "V2", **v2_results}
    ])

    comparison_table.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    print(comparison_table.to_string(index=False))

    print(
        "\nV2 - V1 accuracy difference: "
        f"{comparison['accuracy_difference_percentage_points']:+.4f} "
        "percentage points"
    )

    print(
        "V2 - V1 macro F1 difference: "
        f"{comparison['macro_f1_difference_percentage_points']:+.4f} "
        "percentage points"
    )

    print("\nReports saved in:")
    print(OUTPUT_DIR.relative_to(ROOT))

    print("\nTesting dataset was NOT accessed.")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nCOMPARISON FAILED:", error)
        sys.exit(1)

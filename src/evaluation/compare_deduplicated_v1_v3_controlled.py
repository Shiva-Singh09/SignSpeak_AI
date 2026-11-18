
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

V1_PATH = (
    ROOT
    / "models"
    / "signspeak_mlp_deduplicated_v1.pth"
)

V3_PATH = (
    ROOT
    / "models"
    / "signspeak_mlp_deduplicated_v3_controlled.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "reports"
    / "model_comparison_v1_v3_controlled"
)

COORDINATE_FEATURES = 126
TOTAL_FEATURES = 132
BATCH_SIZE = 512

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

EXPECTED_MAPPING = {
    letter: index
    for index, letter in enumerate(CLASSES)
}


# ============================================================
# EXACT MODEL ARCHITECTURE
# ============================================================

class V1Architecture(nn.Module):
    """
    Architecture shared by V1 and V3.

    Linear -> ReLU -> BatchNorm -> Dropout
    Linear -> ReLU -> BatchNorm -> Dropout
    Linear

    Input:  132 features
    Hidden: 256, 128
    Output: 26 alphabet classes

    Parameters: 71,066
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


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


def prepare_model(checkpoint, version, device):

    if version == "V1":

        model = V1Architecture()

        class_to_idx = checkpoint["class_to_idx"]

        mean = np.asarray(
            checkpoint["scaler_mean"],
            dtype=np.float32
        )

        std = np.asarray(
            checkpoint["scaler_std"],
            dtype=np.float32
        )

    elif version == "V3":

        model = V1Architecture()

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

    else:
        raise ValueError(
            f"Unknown model version: {version}"
        )

    # --------------------------------------------------------
    # VALIDATE CLASS MAPPING
    # --------------------------------------------------------

    if class_to_idx != EXPECTED_MAPPING:
        raise ValueError(
            f"{version}: Class mapping does not match A-Z order"
        )

    # --------------------------------------------------------
    # VALIDATE NORMALIZATION STATISTICS
    # --------------------------------------------------------

    if mean.shape != (COORDINATE_FEATURES,):
        raise ValueError(
            f"{version}: Invalid normalization mean shape: {mean.shape}"
        )

    if std.shape != (COORDINATE_FEATURES,):
        raise ValueError(
            f"{version}: Invalid normalization std shape: {std.shape}"
        )

    if not np.isfinite(mean).all():
        raise ValueError(
            f"{version}: Non-finite normalization mean"
        )

    if not np.isfinite(std).all():
        raise ValueError(
            f"{version}: Non-finite normalization std"
        )

    if np.any(std <= 0):
        raise ValueError(
            f"{version}: Standard deviation must be positive"
        )

    # --------------------------------------------------------
    # STRICT ARCHITECTURE VALIDATION
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(device)
    model.eval()

    return model, class_to_idx, mean, std


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_feature_columns(feature_columns, checkpoint, version):

    if len(feature_columns) != TOTAL_FEATURES:
        raise ValueError(
            f"Expected {TOTAL_FEATURES} features, "
            f"got {len(feature_columns)}"
        )

    # V3 stores its feature column names.
    # V1 checkpoint does not store the complete feature list.
    if "feature_columns" in checkpoint:

        stored_columns = checkpoint["feature_columns"]

        if list(feature_columns) != list(stored_columns):
            raise ValueError(
                f"{version}: Feature ordering mismatch"
            )


# ============================================================
# PREDICTION
# ============================================================

def predict(model, X, mean, std, device):

    X = X.copy()

    # Normalize coordinate features only.
    # Presence and handedness flags remain unchanged.

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

            probs = torch.softmax(
                logits,
                dim=1
            )

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

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

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

    # --------------------------------------------------------
    # SAVE CLASS REPORT
    # --------------------------------------------------------

    pd.DataFrame(
        report
    ).transpose().to_csv(
        OUTPUT_DIR / f"{name.lower()}_class_report.csv"
    )

    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX
    # --------------------------------------------------------

    pd.DataFrame(
        confusion,
        index=classes,
        columns=classes
    ).to_csv(
        OUTPUT_DIR / f"{name.lower()}_confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({

        "image_name": image_names,

        "actual": [
            classes[i]
            for i in y_true
        ],

        "predicted": [
            classes[i]
            for i in y_pred
        ],

        "correct": (
            y_true == y_pred
        ),

        "top1_confidence": (
            probabilities.max(axis=1)
        ),

        "top2_confidence": (
            np.sort(
                probabilities,
                axis=1
            )[:, -2]
        ),

        "confidence_margin": (
            np.sort(
                probabilities,
                axis=1
            )[:, -1]
            -
            np.sort(
                probabilities,
                axis=1
            )[:, -2]
        )
    })

    prediction_df.to_csv(
        OUTPUT_DIR / f"{name.lower()}_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(f"\n{name} RESULTS")
    print("-" * 55)

    print(
        f"Accuracy:         {accuracy * 100:.4f}%"
    )

    print(
        f"Macro precision:  {precision * 100:.4f}%"
    )

    print(
        f"Macro recall:     {recall * 100:.4f}%"
    )

    print(
        f"Macro F1:         {f1 * 100:.4f}%"
    )

    print("\nWorst 5 classes by F1:")

    class_scores = []

    for letter in classes:

        class_scores.append(
            (
                letter,
                report[letter]["f1-score"],
                report[letter]["precision"],
                report[letter]["recall"],
                int(report[letter]["support"])
            )
        )

    class_scores.sort(
        key=lambda item: item[1]
    )

    print(
        f"{'Class':<8}"
        f"{'F1':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'Support':<10}"
    )

    for letter, class_f1, class_precision, class_recall, support in class_scores[:5]:

        print(
            f"{letter:<8}"
            f"{class_f1 * 100:<12.2f}"
            f"{class_precision * 100:<12.2f}"
            f"{class_recall * 100:<12.2f}"
            f"{support:<10}"
        )

    incorrect = int(
        (y_true != y_pred).sum()
    )

    print(
        "\nMisclassified samples:",
        incorrect
    )

    return {

        "accuracy": float(accuracy),

        "macro_precision": float(precision),

        "macro_recall": float(recall),

        "macro_f1": float(f1),

        "correct": int(
            (y_true == y_pred).sum()
        ),

        "incorrect": incorrect,

        "samples": int(
            len(y_true)
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)

    print(
        "SIGNSPEAK - V1 VS V3 CONTROLLED "
        "HAND-PRESENT VALIDATION COMPARISON"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    print(
        "\n[1/4] LOADING COMMON VALIDATION DATA"
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    feature_columns = [
        col
        for col in df.columns
        if col not in ("label", "image_name")
    ]

    if len(feature_columns) != TOTAL_FEATURES:
        raise ValueError(
            f"Expected {TOTAL_FEATURES} features, "
            f"got {len(feature_columns)}"
        )

    if "image_name" not in df.columns:
        raise ValueError(
            "image_name column missing"
        )

    if "label" not in df.columns:
        raise ValueError(
            "label column missing"
        )

    if df["label"].isna().any():
        raise ValueError(
            "Missing validation labels"
        )

    # --------------------------------------------------------
    # VALIDATE LABELS
    # --------------------------------------------------------

    invalid_labels = (
        set(df["label"].astype(str).unique())
        -
        set(CLASSES)
    )

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels: {invalid_labels}"
        )

    # --------------------------------------------------------
    # VALIDATE FEATURES
    # --------------------------------------------------------

    X = df[feature_columns].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X).all():
        raise ValueError(
            "Non-finite validation features"
        )

    if "hand_0_present" not in df.columns:
        raise ValueError(
            "hand_0_present feature missing"
        )

    if "hand_1_present" not in df.columns:
        raise ValueError(
            "hand_1_present feature missing"
        )

    hand_0 = pd.to_numeric(
        df["hand_0_present"],
        errors="raise"
    )

    hand_1 = pd.to_numeric(
        df["hand_1_present"],
        errors="raise"
    )

    no_hand = int(
        (
            (hand_0 == 0)
            &
            (hand_1 == 0)
        ).sum()
    )

    if no_hand != 0:
        raise ValueError(
            f"Expected hand-present validation data, "
            f"found {no_hand} no-hand rows"
        )

    invalid_presence = (
        ~hand_0.isin([0, 1])
        |
        ~hand_1.isin([0, 1])
        |
        ((hand_0 == 0) & (hand_1 == 0))
    )

    if invalid_presence.any():
        raise ValueError(
            "Invalid hand-presence flags detected"
        )

    image_names = (
        df["image_name"]
        .astype(str)
        .to_numpy()
    )

    # --------------------------------------------------------
    # LOAD CHECKPOINTS
    # --------------------------------------------------------

    print(
        "\n[2/4] LOADING CHECKPOINTS"
    )

    v1_checkpoint = load_checkpoint(
        V1_PATH
    )

    v3_checkpoint = load_checkpoint(
        V3_PATH
    )

    # Validate V3 checkpoint metadata.
    if v3_checkpoint.get("experiment") != "deduplicated_v3_controlled":
        raise ValueError(
            "V3 checkpoint experiment metadata mismatch"
        )

    if v3_checkpoint.get("input_size") != TOTAL_FEATURES:
        raise ValueError(
            "V3 checkpoint input size mismatch"
        )

    if v3_checkpoint.get("num_classes") != len(CLASSES):
        raise ValueError(
            "V3 checkpoint class count mismatch"
        )

    # Prepare models.
    v1, v1_mapping, v1_mean, v1_std = prepare_model(
        v1_checkpoint,
        "V1",
        device
    )

    v3, v3_mapping, v3_mean, v3_std = prepare_model(
        v3_checkpoint,
        "V3",
        device
    )

    # Verify feature ordering.
    validate_feature_columns(
        feature_columns,
        v1_checkpoint,
        "V1"
    )

    validate_feature_columns(
        feature_columns,
        v3_checkpoint,
        "V3"
    )

    # --------------------------------------------------------
    # VALIDATE ARCHITECTURE AND MAPPING
    # --------------------------------------------------------

    if v1_mapping != v3_mapping:
        raise ValueError(
            "V1 and V3 class mappings do not match"
        )

    y_true = np.array(
        [
            EXPECTED_MAPPING[str(label)]
            for label in df["label"]
        ],
        dtype=np.int64
    )

    print(
        "Validation samples:",
        len(df)
    )

    print(
        "Features:",
        len(feature_columns)
    )

    print(
        "Classes:",
        len(CLASSES)
    )

    print(
        "No-hand rows:",
        no_hand
    )

    print(
        "V1/V3 architecture:",
        "verified through strict state loading"
    )

    print(
        "Class mapping:",
        "verified"
    )

    print(
        "Feature ordering:",
        "verified"
    )

    print(
        "V1 best epoch:",
        v1_checkpoint.get("best_epoch", "Not stored")
    )

    print(
        "V3 best epoch:",
        v3_checkpoint.get("best_epoch", "Not stored")
    )

    # --------------------------------------------------------
    # EVALUATE BOTH MODELS
    # --------------------------------------------------------

    print(
        "\n[3/4] EVALUATING BOTH MODELS"
    )

    v1_results = evaluate_model(
        "V1",
        v1,
        X,
        y_true,
        CLASSES,
        v1_mean,
        v1_std,
        device,
        image_names
    )

    v3_results = evaluate_model(
        "V3",
        v3,
        X,
        y_true,
        CLASSES,
        v3_mean,
        v3_std,
        device,
        image_names
    )

    # --------------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------------

    print(
        "\n[4/4] SAVING COMPARISON"
    )

    accuracy_difference = (
        v3_results["accuracy"]
        -
        v1_results["accuracy"]
    ) * 100

    f1_difference = (
        v3_results["macro_f1"]
        -
        v1_results["macro_f1"]
    ) * 100

    comparison = {

        "validation_samples": len(df),

        "validation_dataset": "hand_present",

        "v1": v1_results,

        "v3": v3_results,

        "accuracy_difference_percentage_points": (
            accuracy_difference
        ),

        "macro_f1_difference_percentage_points": (
            f1_difference
        ),

        "experiment_note": (
            "V1 and V3 use the same MLP architecture, "
            "feature ordering and nominal training configuration. "
            "V1 was trained using the original training dataset "
            "including no-hand samples. V3 was trained using "
            "hand-present samples only. Both models are evaluated "
            "on the exact same hand-present validation samples. "
            "Each model uses its own training-derived normalization "
            "statistics. This comparison evaluates the effect of "
            "the changed training data pipeline, although other "
            "training-run variability cannot be ruled out."
        )
    }

    with open(
        OUTPUT_DIR / "comparison_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            comparison,
            file,
            indent=4
        )

    comparison_table = pd.DataFrame([
        {
            "Model": "V1",
            **v1_results
        },
        {
            "Model": "V3",
            **v3_results
        }
    ])

    comparison_table.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "FINAL V1 VS V3 CONTROLLED COMPARISON"
    )

    print("=" * 70)

    print(
        comparison_table.to_string(
            index=False
        )
    )

    print(
        "\nV3 - V1 accuracy difference: "
        f"{accuracy_difference:+.4f} "
        "percentage points"
    )

    print(
        "V3 - V1 macro F1 difference: "
        f"{f1_difference:+.4f} "
        "percentage points"
    )

    if accuracy_difference > 0:
        print(
            "\nV3 has higher validation accuracy."
        )

    elif accuracy_difference < 0:
        print(
            "\nV1 has higher validation accuracy."
        )

    else:
        print(
            "\nBoth models have equal validation accuracy."
        )

    if f1_difference > 0:
        print(
            "V3 has higher macro F1."
        )

    elif f1_difference < 0:
        print(
            "V1 has higher macro F1."
        )

    else:
        print(
            "Both models have equal macro F1."
        )

    print(
        "\nReports saved in:"
    )

    print(
        OUTPUT_DIR.relative_to(ROOT)
    )

    print(
        "\nTesting dataset was NOT accessed."
    )

    print(
        "\nCOMPARISON COMPLETED SUCCESSFULLY"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(
            "\nCOMPARISON FAILED:",
            error
        )

        sys.exit(1)

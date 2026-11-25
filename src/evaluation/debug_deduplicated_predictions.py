
import torch
import numpy as np
from pathlib import Path

from analyze_deduplicated_v1 import (
    SignSpeakMLP,
    load_validation_data,
    FEATURE_FILE,
)

CHECKPOINT_PATH = Path(
    "models/signspeak_mlp_deduplicated_v1.pth"
)


def main():

    print("=" * 60)
    print("DEBUGGING DEDUPLICATED V1 PREDICTIONS")
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

    X, y_true, image_names = load_validation_data(
        class_to_idx
    )

    mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        checkpoint["scaler_std"],
        dtype=np.float32,
    )

    X[:, :126] = (X[:, :126] - mean) / std

    model = SignSpeakMLP()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            torch.tensor(X, dtype=torch.float32)
        )

        predictions = logits.argmax(dim=1).numpy()

    print(f"\nValidation samples: {len(y_true)}")

    unique, counts = np.unique(
        predictions,
        return_counts=True,
    )

    print("\nPREDICTED CLASS DISTRIBUTION")
    print("-" * 45)

    for index, count in zip(unique, counts):

        label = idx_to_class[int(index)]

        print(f"{label}: {count}")

    d_index = class_to_idx["D"]

    print("\nD CLASS CHECK")
    print("-" * 45)

    print(f"D class index: {d_index}")
    print(f"Predictions as D: {np.sum(predictions == d_index)}")
    print(f"Actual D samples: {np.sum(y_true == d_index)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

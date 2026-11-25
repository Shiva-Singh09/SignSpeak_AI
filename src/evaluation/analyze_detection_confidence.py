
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    ROOT / "dataset" / "processed"
    / "realsign_clean" / "training_landmarks.csv"
)

CLASSES = ["Q", "N", "W", "M", "V", "B"]


def main():
    data = pd.read_csv(CSV_PATH)

    print("\nMEDIAPIPE DETECTION CONFIDENCE")
    print("=" * 75)

    for label in CLASSES:
        subset = data[data["label"].astype(str) == label]

        print(f"\nCLASS: {label} | Samples: {len(subset)}")

        for hand in ("hand_0", "hand_1"):
            category = subset[f"{hand}_category"].astype(str)
            confidence = pd.to_numeric(
                subset[f"{hand}_confidence"],
                errors="coerce",
            )

            detected = category.isin(["Left", "Right"])
            valid_confidence = confidence[detected].dropna()

            print(f"\n  {hand}:")
            print(f"    Detected: {int(detected.sum())}")
            print(f"    Missing : {int((~detected).sum())}")

            if len(valid_confidence):
                print(
                    f"    Mean confidence: "
                    f"{valid_confidence.mean():.4f}"
                )
                print(
                    f"    Median confidence: "
                    f"{valid_confidence.median():.4f}"
                )
                print(
                    f"    Minimum confidence: "
                    f"{valid_confidence.min():.4f}"
                )


if __name__ == "__main__":
    main()

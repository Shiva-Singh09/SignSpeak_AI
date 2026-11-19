
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

    print("\nHANDEDNESS DISTRIBUTION")
    print("=" * 70)

    for label in CLASSES:
        subset = data[data["label"].astype(str) == label]

        print(f"\nCLASS: {label}")

        for hand in ("hand_0", "hand_1"):
            categories = (
                subset[hand + "_category"]
                .replace("", "Missing")
                .fillna("Missing")
            )

            print(f"\n{hand}:")
            print(
                categories.value_counts().to_string()
            )

        print("\nHand 0 + Hand 1 combinations:")

        combinations = (
            subset["hand_0_category"].fillna("Missing")
            .replace("", "Missing")
            + " + "
            + subset["hand_1_category"].fillna("Missing")
            .replace("", "Missing")
        )

        print(combinations.value_counts().to_string())


if __name__ == "__main__":
    main()

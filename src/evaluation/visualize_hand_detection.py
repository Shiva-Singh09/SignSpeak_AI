
from pathlib import Path
import random

import cv2
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    ROOT / "dataset" / "processed"
    / "realsign_clean" / "training_landmarks.csv"
)

IMAGE_ROOT = (
    ROOT / "dataset" / "external"
    / "RealSign" / "extracted" / "Training"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "reports"
    / "hand_detection_visual_audit.jpg"
)

CLASSES = ["Q", "N", "W", "M", "V", "B"]
SAMPLES_PER_GROUP = 3
RANDOM_SEED = 42

# MediaPipe's 21 hand-landmark connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand(image, row, prefix, color):
    height, width = image.shape[:2]
    points = []

    for index in range(21):
        x = float(row[f"{prefix}_{index:02d}_x"])
        y = float(row[f"{prefix}_{index:02d}_y"])

        px = int(x * width)
        py = int(y * height)
        points.append((px, py))

    # Draw connections
    for start, end in CONNECTIONS:
        cv2.line(
            image,
            points[start],
            points[end],
            color,
            2,
            cv2.LINE_AA,
        )

    # Draw landmarks
    for point in points:
        cv2.circle(image, point, 4, color, -1)


def create_overlay(image, row):
    overlay = image.copy()

    hand_0 = str(row["hand_0_category"]).strip()
    hand_1 = str(row["hand_1_category"]).strip()

    if hand_0 in ("Left", "Right"):
        draw_hand(
            overlay, row, "hand_0", (0, 255, 0)
        )

    if hand_1 in ("Left", "Right"):
        draw_hand(
            overlay, row, "hand_1", (255, 100, 0)
        )

    return overlay


def main():
    random.seed(RANDOM_SEED)

    data = pd.read_csv(CSV_PATH)

    samples = []

    for label in CLASSES:
        class_data = data[data["label"].astype(str) == label]

        for hand_count in (1, 2):
            group = class_data[
                class_data["detected_hands"] == hand_count
            ]

            if group.empty:
                print(
                    f"WARNING: No {hand_count}-hand samples "
                    f"for class {label}"
                )
                continue

            selected = group.sample(
                n=min(SAMPLES_PER_GROUP, len(group)),
                random_state=RANDOM_SEED,
            )

            for _, row in selected.iterrows():
                samples.append((label, hand_count, row))

    if not samples:
        raise RuntimeError("No samples found.")

    fig, axes = plt.subplots(
        len(samples), 2,
        figsize=(12, max(5, len(samples) * 2.3)),
        squeeze=False,
    )

    for index, (label, hand_count, row) in enumerate(samples):
        filename = Path(str(row["image_name"])).name

        image_path = IMAGE_ROOT / label / filename

        original = cv2.imread(str(image_path))

        if original is None:
            axes[index, 0].text(
                0.5, 0.5,
                f"Image not found:\n{filename}",
                ha="center",
                va="center",
            )
            axes[index, 1].axis("off")
            continue

        original = cv2.cvtColor(
            original, cv2.COLOR_BGR2RGB
        )

        overlay = create_overlay(original, row)

        axes[index, 0].imshow(original)
        axes[index, 0].set_title(
            f"{label} | {hand_count} detected hand(s)\n"
            f"{filename} | Original",
            fontsize=8,
        )

        axes[index, 1].imshow(overlay)
        axes[index, 1].set_title(
            f"Landmark overlay\n"
            f"Hand 0: {row['hand_0_category']} | "
            f"Hand 1: {row['hand_1_category']}",
            fontsize=8,
        )

        axes[index, 0].axis("off")
        axes[index, 1].axis("off")

    fig.suptitle(
        "RealSign: Original Images vs MediaPipe Detection",
        fontsize=15,
    )

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nVISUAL AUDIT CREATED")
    print("=" * 55)
    print(f"Images inspected: {len(samples)}")
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Samples per group: up to {SAMPLES_PER_GROUP}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

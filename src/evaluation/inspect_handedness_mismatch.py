
from pathlib import Path
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

OUTPUT = (
    ROOT / "outputs" / "reports"
    / "handedness_mismatch_audit.jpg"
)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand(image, row, prefix, color):
    h, w = image.shape[:2]
    points = []

    for i in range(21):
        x = int(float(row[f"{prefix}_{i:02d}_x"]) * w)
        y = int(float(row[f"{prefix}_{i:02d}_y"]) * h)
        points.append((x, y))

    for a, b in CONNECTIONS:
        cv2.line(image, points[a], points[b], color, 2)

    for point in points:
        cv2.circle(image, point, 3, color, -1)


def main():
    data = pd.read_csv(CSV_PATH)

    groups = [
        ("M", "Left", "Left"),
        ("M", "Right", "Left"),
        ("W", "Right", "Right"),
        ("W", "Left", "Right"),
    ]

    samples = []

    for label, category0, category1 in groups:
        subset = data[
            (data["label"] == label)
            & (data["detected_hands"] == 2)
            & (data["hand_0_category"] == category0)
            & (data["hand_1_category"] == category1)
        ]

        if subset.empty:
            print("No samples:", label, category0, category1)
            continue

        selected = subset.sample(
            n=min(3, len(subset)),
            random_state=42,
        )

        for _, row in selected.iterrows():
            samples.append((label, category0, category1, row))

    if not samples:
        raise RuntimeError("No matching images found.")

    fig, axes = plt.subplots(
        len(samples), 2,
        figsize=(12, max(6, len(samples) * 2.5)),
        squeeze=False,
    )

    for index, (label, cat0, cat1, row) in enumerate(samples):
        filename = Path(str(row["image_name"])).name
        image_path = IMAGE_ROOT / label / filename

        image = cv2.imread(str(image_path))

        if image is None:
            axes[index, 0].text(0.5, 0.5, "Image missing",
                                ha="center")
            axes[index, 1].axis("off")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = image.copy()

        draw_hand(overlay, row, "hand_0", (0, 255, 0))
        draw_hand(overlay, row, "hand_1", (255, 100, 0))

        confidence0 = row["hand_0_confidence"]
        confidence1 = row["hand_1_confidence"]

        axes[index, 0].imshow(image)
        axes[index, 0].set_title(
            f"{label} | Original | {filename}", fontsize=8
        )

        axes[index, 1].imshow(overlay)
        axes[index, 1].set_title(
            f"Hand 0: {cat0} ({confidence0}) | "
            f"Hand 1: {cat1} ({confidence1})",
            fontsize=8,
        )

        axes[index, 0].axis("off")
        axes[index, 1].axis("off")

    fig.suptitle(
        "MediaPipe Handedness Consistency Audit",
        fontsize=14,
    )

    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("\nAudit created successfully.")
    print("Samples:", len(samples))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()

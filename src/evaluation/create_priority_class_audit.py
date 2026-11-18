
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    ROOT / "dataset" / "external"
    / "RealSign" / "extracted" / "Training"
)

OUTPUT = (
    ROOT / "outputs" / "reports"
    / "priority_class_visual_audit.jpg"
)

# Difficult classes and their most common confusion partners.
CLASSES = [
    "Q", "N",
    "S", "Q",
    "W", "M",
    "N", "M",
    "V", "F",
    "Z", "B",
]

IMAGES_PER_CLASS = 8
RANDOM_SEED = 42


def main():

    random.seed(RANDOM_SEED)

    rows = len(CLASSES)
    cols = IMAGES_PER_CLASS

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, rows * 2.0)
    )

    fig.suptitle(
        "RealSign Training Images - Priority Class Audit",
        fontsize=16
    )

    for row_index, label in enumerate(CLASSES):

        class_folder = DATASET / label

        if not class_folder.exists():
            raise FileNotFoundError(class_folder)

        files = sorted([
            path for path in class_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in (
                ".jpg", ".jpeg", ".png", ".bmp"
            )
        ])

        if not files:
            raise RuntimeError(
                f"No images found for class {label}"
            )

        selected = random.sample(
            files,
            min(IMAGES_PER_CLASS, len(files))
        )

        for col_index in range(cols):

            ax = axes[row_index, col_index]
            ax.axis("off")

            if col_index >= len(selected):
                continue

            image_path = selected[col_index]

            image = cv2.imread(str(image_path))

            if image is None:
                ax.set_title("Unreadable")
                continue

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            ax.imshow(image)

            ax.set_title(
                f"{label} | {image_path.name}",
                fontsize=7
            )

        axes[row_index, 0].set_ylabel(
            f"Folder label: {label}",
            fontsize=9
        )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUTPUT,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Visual audit created successfully.")
    print(f"Output: {OUTPUT.relative_to(ROOT)}")
    print(f"Classes inspected: {len(CLASSES)}")
    print(f"Images per class: up to {IMAGES_PER_CLASS}")


if __name__ == "__main__":
    main()

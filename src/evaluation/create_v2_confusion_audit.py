
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

IMAGE_ROOT = (
    ROOT / "dataset" / "external"
    / "RealSign" / "extracted" / "Validation"
)

OUTPUT = (
    ROOT / "outputs" / "reports"
    / "v2_confusion_visual_audit.jpg"
)

random.seed(42)

# Actual and predicted confusion pairs from V2.
PAIRS = [
    ("Q", "N"),
    ("S", "Q"),
    ("Y", "U"),
    ("W", "M"),
    ("Z", "P"),
]

SAMPLES_PER_CLASS = 4


def main():

    classes = []

    for actual, predicted in PAIRS:
        classes.extend([actual, predicted])

    classes = list(dict.fromkeys(classes))

    fig, axes = plt.subplots(
        len(classes),
        SAMPLES_PER_CLASS,
        figsize=(12, 18),
        squeeze=False
    )

    print("Creating visual audit...")

    for row_index, letter in enumerate(classes):

        folder = IMAGE_ROOT / letter

        images = sorted(folder.glob("*.jpg"))

        if not images:
            print("No images found:", letter)
            continue

        selected = random.sample(
            images,
            min(SAMPLES_PER_CLASS, len(images))
        )

        for col_index in range(SAMPLES_PER_CLASS):

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
                image, cv2.COLOR_BGR2RGB
            )

            ax.imshow(image)

            ax.set_title(
                image_path.name,
                fontsize=8
            )

        axes[row_index, 0].set_ylabel(
            f"CLASS {letter}",
            fontsize=11,
            rotation=0,
            labelpad=35
        )

    fig.suptitle(
        "V2 Confusing Alphabet Classes - Visual Audit",
        fontsize=15
    )

    plt.tight_layout()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nVISUAL AUDIT CREATED")
    print("Classes:", ", ".join(classes))
    print("Images:", len(classes) * SAMPLES_PER_CLASS)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()

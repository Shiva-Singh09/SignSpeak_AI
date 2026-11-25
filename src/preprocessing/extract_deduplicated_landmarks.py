
from pathlib import Path
import csv
import json
from collections import Counter

import cv2
import mediapipe as mp


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT
    / "dataset"
    / "processed"
    / "realsign_deduplicated"
    / "images"
)

MODEL = PROJECT / "models" / "hand_landmarker.task"

OUTPUT = (
    PROJECT
    / "dataset"
    / "processed"
    / "realsign_deduplicated"
    / "landmarks"
)

REPORTS = OUTPUT / "reports"

SPLITS = ["Training", "Validation", "Testing"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ==================================================
# 2. VALIDATE PATHS
# ==================================================

if not MODEL.exists():
    raise FileNotFoundError(f"Model not found: {MODEL}")

if not DATA_ROOT.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_ROOT}")

for split in SPLITS:
    if not (DATA_ROOT / split).exists():
        raise FileNotFoundError(
            f"Missing split folder: {DATA_ROOT / split}"
        )

OUTPUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


# ==================================================
# 3. CSV COLUMNS
# ==================================================

metadata = [
    "label",
    "split",
    "image_name",
    "detected_hands",
    "hand_0_category",
    "hand_0_confidence",
    "hand_1_category",
    "hand_1_confidence",
    "detection_status"
]

landmark_columns = []

for hand in range(2):
    for point in range(21):
        for axis in ("x", "y", "z"):
            landmark_columns.append(
                f"hand_{hand}_{point:02d}_{axis}"
            )

fieldnames = metadata + landmark_columns


# ==================================================
# 4. MEDIAPIPE CONFIGURATION
# ==================================================

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5
)


# ==================================================
# 5. EXTRACTION STATISTICS
# ==================================================

summary = {}

total_processed = 0
total_detected = 0
total_no_hand = 0
total_unreadable = 0

hand_counts = Counter()


# ==================================================
# 6. LANDMARK EXTRACTION
# ==================================================

print("=" * 65)
print("REALSign DEDUPLICATED DATASET - LANDMARK EXTRACTION")
print("=" * 65)

with mp.tasks.vision.HandLandmarker.create_from_options(
    options
) as detector:

    for split in SPLITS:

        split_root = DATA_ROOT / split

        output_file = (
            OUTPUT / f"{split.lower()}_landmarks.csv"
        )

        processed = 0
        detected = 0
        no_hand = 0
        unreadable = 0

        split_hand_counts = Counter()

        print(f"\nProcessing {split}...")

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for letter in LETTERS:

                class_dir = split_root / letter

                if not class_dir.exists():
                    raise FileNotFoundError(
                        f"Missing class folder: {class_dir}"
                    )

                image_files = sorted(
                    path for path in class_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                )

                for image_path in image_files:

                    image = cv2.imread(str(image_path))

                    row = {
                        "label": letter,
                        "split": split,
                        "image_name": str(
                            image_path.relative_to(DATA_ROOT)
                        ),
                        "detected_hands": 0,
                        "hand_0_category": "",
                        "hand_0_confidence": "",
                        "hand_1_category": "",
                        "hand_1_confidence": "",
                        "detection_status": "ok"
                    }

                    # Initialize all 126 coordinates.
                    for column in landmark_columns:
                        row[column] = 0.0

                    if image is None:

                        row["detection_status"] = "unreadable"
                        unreadable += 1
                        processed += 1

                        writer.writerow(row)
                        continue

                    rgb = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    result = detector.detect(mp_image)

                    count = len(result.hand_landmarks)

                    row["detected_hands"] = count

                    split_hand_counts[count] += 1
                    hand_counts[count] += 1

                    if count == 0:
                        no_hand += 1
                    else:
                        detected += 1

                    for hand_index, landmarks in enumerate(
                        result.hand_landmarks[:2]
                    ):

                        category = (
                            result.handedness[hand_index][0]
                        )

                        row[
                            f"hand_{hand_index}_category"
                        ] = category.category_name

                        row[
                            f"hand_{hand_index}_confidence"
                        ] = round(category.score, 5)

                        for point_index, landmark in enumerate(
                            landmarks
                        ):

                            row[
                                f"hand_{hand_index}_{point_index:02d}_x"
                            ] = landmark.x

                            row[
                                f"hand_{hand_index}_{point_index:02d}_y"
                            ] = landmark.y

                            row[
                                f"hand_{hand_index}_{point_index:02d}_z"
                            ] = landmark.z

                    writer.writerow(row)

                    processed += 1

                    if processed % 500 == 0:
                        print(
                            f"{split}: {processed} images processed"
                        )

        # ------------------------------------------
        # Split summary
        # ------------------------------------------

        total_processed += processed
        total_detected += detected
        total_no_hand += no_hand
        total_unreadable += unreadable

        summary[split] = {
            "images_processed": processed,
            "images_with_hands": detected,
            "images_without_hands": no_hand,
            "unreadable_images": unreadable,
            "hand_count_distribution": {
                str(key): value
                for key, value in sorted(
                    split_hand_counts.items()
                )
            }
        }

        print(f"\n{split} completed")
        print("Images processed:", processed)
        print("Images with hands:", detected)
        print("Images without hands:", no_hand)
        print("Unreadable images:", unreadable)
        print("Hand distribution:", dict(split_hand_counts))
        print("CSV saved:", output_file)


# ==================================================
# 7. SAVE EXTRACTION REPORT
# ==================================================

summary["overall"] = {
    "total_images_processed": total_processed,
    "images_with_hands": total_detected,
    "images_without_hands": total_no_hand,
    "unreadable_images": total_unreadable,
    "hand_count_distribution": {
        str(key): value
        for key, value in sorted(hand_counts.items())
    }
}

report_file = REPORTS / "landmark_extraction_summary.json"

with report_file.open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)


# ==================================================
# 8. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("FULL EXTRACTION SUMMARY")
print("=" * 65)

print("Total images processed:", total_processed)
print("Images with detected hands:", total_detected)
print("Images without detected hands:", total_no_hand)
print("Unreadable images:", total_unreadable)
print("Hand-count distribution:", dict(hand_counts))

print("\nOutput folder:", OUTPUT)
print("Report:", report_file)

print("\nLANDMARK EXTRACTION COMPLETED")

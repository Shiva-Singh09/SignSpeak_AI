from pathlib import Path
import csv
import cv2
import mediapipe as mp

PROJECT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT / "dataset" / "external" /
    "RealSign" / "extracted"
)

MODEL = PROJECT / "models" / "hand_landmarker.task"

OUTPUT = (
    PROJECT / "dataset" / "processed" /
    "realsign_landmarks"
)

SPLITS = ["Training", "Validation", "Testing"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

if not MODEL.exists():
    raise FileNotFoundError(f"Model not found: {MODEL}")

OUTPUT.mkdir(parents=True, exist_ok=True)

# CSV metadata columns.
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

# 126 landmark coordinate columns.
landmark_columns = []

for hand in range(2):
    for point in range(21):
        for axis in ("x", "y", "z"):
            landmark_columns.append(
                f"hand_{hand}_{point:02d}_{axis}"
            )

fieldnames = metadata + landmark_columns

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5
)

total_processed = 0
total_detected = 0
total_no_hand = 0
total_unreadable = 0
hand_counts = {0: 0, 1: 0, 2: 0}

with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:

    for split in SPLITS:
        split_root = DATA_ROOT / split
        output_file = OUTPUT / f"{split.lower()}_landmarks.csv"

        processed = 0
        detected = 0
        no_hand = 0
        unreadable = 0

        print(f"\nProcessing {split}...")

        with output_file.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file, fieldnames=fieldnames
            )
            writer.writeheader()

            for letter in LETTERS:
                image_files = sorted(
                    (split_root / letter).glob("*.jpg")
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

                    # Initialize all 126 features to zero.
                    for column in landmark_columns:
                        row[column] = 0.0

                    if image is None:
                        row["detection_status"] = "unreadable"
                        unreadable += 1
                        writer.writerow(row)
                        processed += 1
                        continue

                    rgb = cv2.cvtColor(
                        image, cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    result = detector.detect(mp_image)

                    count = len(result.hand_landmarks)
                    row["detected_hands"] = count

                    if count == 0:
                        no_hand += 1
                    else:
                        detected += 1

                    hand_counts[count] = (
                        hand_counts.get(count, 0) + 1
                    )

                    for hand_index, landmarks in enumerate(
                        result.hand_landmarks[:2]
                    ):
                        category = result.handedness[hand_index][0]

                        row[f"hand_{hand_index}_category"] = (
                            category.category_name
                        )
                        row[f"hand_{hand_index}_confidence"] = (
                            round(category.score, 5)
                        )

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

                    if processed % 250 == 0:
                        print(
                            f"{split}: {processed} images processed"
                        )

        total_processed += processed
        total_detected += detected
        total_no_hand += no_hand
        total_unreadable += unreadable

        print(f"\n{split} completed")
        print("Images processed:", processed)
        print("Images with hands:", detected)
        print("Images without hands:", no_hand)
        print("Unreadable images:", unreadable)
        print("CSV saved:", output_file)

print("\n" + "=" * 45)
print("FULL EXTRACTION SUMMARY")
print("=" * 45)
print("Total images processed:", total_processed)
print("Images with detected hands:", total_detected)
print("Images without detected hands:", total_no_hand)
print("Unreadable images:", total_unreadable)
print("Hand-count distribution:", hand_counts)
print("Output folder:", OUTPUT)

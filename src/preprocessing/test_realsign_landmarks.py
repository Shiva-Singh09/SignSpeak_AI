from pathlib import Path
import csv
import cv2
import mediapipe as mp
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "dataset" / "external" / "RealSign" / "extracted" / "Training"
MODEL = PROJECT / "models" / "hand_landmarker.task"
OUTPUT = PROJECT / "outputs" / "reports" / "realsign_pilot"

OUTPUT.mkdir(parents=True, exist_ok=True)

if not MODEL.exists():
    raise FileNotFoundError(f"Hand Landmarker model not found: {MODEL}")

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# MediaPipe hand skeleton connections.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5
)

rows = []
columns = 5
cell_w, cell_h = 240, 270
sheet = np.full((6 * cell_h, columns * cell_w, 3), 245, dtype=np.uint8)

with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:

    for index, letter in enumerate(letters):
        files = sorted((DATASET / letter).glob("*.jpg"))

        if not files:
            print(f"{letter}: NO IMAGE FOUND")
            continue

        # Use the same representative-image selection as our contact sheet.
        image_path = files[len(files) // 2]
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"{letter}: IMAGE UNREADABLE")
            continue

        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        hand_count = len(result.hand_landmarks)
        handedness_names = []
        confidence_scores = []

        for hand_index, landmarks in enumerate(result.hand_landmarks):
            category = result.handedness[hand_index][0]
            handedness_names.append(category.category_name)
            confidence_scores.append(category.score)

           # Draw every detected hand on the preview image.
            points = [
                (int(lm.x * width), int(lm.y * height))
                for lm in landmarks
            ]

            for start, end in CONNECTIONS:
                cv2.line(
                    image, points[start], points[end],
                    (0, 220, 0), 2
                )

            for point in points:
                cv2.circle(image, point, 4, (0, 0, 255), -1)

        row = {
            "letter": letter,
            "image": str(image_path.relative_to(DATASET)),
            "detected_hands": hand_count,
            "landmarks_per_hand": (
                len(result.hand_landmarks[0])
                if hand_count else 0
            ),
            "handedness": ";".join(handedness_names),
            "confidence": ";".join(
                f"{score:.4f}" for score in confidence_scores
            )
        }
        rows.append(row)

        # Prepare annotated image for contact sheet.
        scale = min(210 / width, 210 / height)
        resized = cv2.resize(
            image,
            (int(width * scale), int(height * scale))
        )

        x = (index % columns) * cell_w
        y = (index // columns) * cell_h

        status = f"{letter}: {hand_count} hand(s)"
        cv2.putText(
            sheet, status, (x + 8, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (20, 20, 20), 2
        )

        rh, rw = resized.shape[:2]
        px = x + (cell_w - rw) // 2
        py = y + 35 + (210 - rh) // 2
        sheet[py:py + rh, px:px + rw] = resized

        print(
            f"{letter}: hands={hand_count}, "
            f"landmarks={row['landmarks_per_hand']}, "
            f"handedness={row['handedness']}, "
            f"confidence={row['confidence']}"
        )

csv_path = OUTPUT / "pilot_detection_report.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

preview_path = OUTPUT / "pilot_landmark_preview.jpg"
cv2.imwrite(str(preview_path), sheet)

successful = sum(
    1 for row in rows
    if row["detected_hands"] > 0
    and row["landmarks_per_hand"] == 21
)

print("\nPILOT SUMMARY")
print("Letters tested:", len(rows))
print("Images with detected hands:", sum(
    1 for row in rows if row["detected_hands"] > 0
))
print("Images with 21 landmarks:", successful)
print("Report:", csv_path)
print("Preview:", preview_path)

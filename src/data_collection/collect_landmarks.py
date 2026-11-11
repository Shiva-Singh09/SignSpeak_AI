
import cv2
import csv
import time
import uuid
from pathlib import Path
from datetime import datetime

import mediapipe as mp


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "raw" / "alphabet_landmarks.csv"

# Select an ISL alphabet letter
LABEL = input("Enter ISL alphabet letter (A-Z): ").strip().upper()

if LABEL not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or len(LABEL) != 1:
    print("ERROR: Please enter one valid letter from A to Z.")
    raise SystemExit(1)

FRAMES_PER_SAMPLE = 30


# --------------------------------------------------
# MEDIAPIPE CONFIGURATION
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# --------------------------------------------------
# CSV STRUCTURE
# --------------------------------------------------

def create_csv_headers():
    headers = [
        "sample_id",
        "label",
        "captured_at",
        "frame_index",
        "hand_count"
    ]

    for hand in ["left", "right"]:
        for landmark in range(21):
            for axis in ["x", "y", "z"]:
                headers.append(f"{hand}_{landmark:02d}_{axis}")

    return headers


def save_sample(rows):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(create_csv_headers())

        writer.writerows(rows)

    print(f"\nSaved {len(rows)} frames to: {OUTPUT_PATH}")


# --------------------------------------------------
# LANDMARK EXTRACTION
# --------------------------------------------------

def extract_frame_data(result):
    hand_data = {
        "left": [0.0] * 63,
        "right": [0.0] * 63
    }

    detected_hands = len(result.hand_landmarks)

    for index, landmarks in enumerate(result.hand_landmarks):
        # MediaPipe handedness is aligned with detected hands.
        # The input frame is mirrored before processing.
        side = "unknown"

        if index < len(result.handedness) and result.handedness[index]:
            side = result.handedness[index][0].category_name.lower()

        if side not in ("left", "right"):
            side = "left" if index == 0 else "right"

        values = []

        for landmark in landmarks:
            values.extend([
                float(landmark.x),
                float(landmark.y),
                float(landmark.z)
            ])

        hand_data[side] = values

    return detected_hands, hand_data


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    if not MODEL_PATH.exists():
        print("ERROR: Hand Landmarker model not found.")
        print(MODEL_PATH)
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    sample_rows = []
    recording = False
    frame_index = 0
    sample_id = None
    captured_at = None
    previous_timestamp = 0

    print("\nSignSpeak AI - Data Collection")
    print("--------------------------------")
    print("Label:", LABEL)
    print("Frames per sample:", FRAMES_PER_SAMPLE)
    print("\nPress S to record a sample.")
    print("Press Q to quit.")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:

                success, frame = camera.read()

                if not success:
                    print("ERROR: Could not read webcam frame.")
                    break

                # Mirror the webcam image.
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = max(
                    previous_timestamp + 1,
                    int(time.monotonic() * 1000)
                )
                previous_timestamp = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                hand_count, hand_data = extract_frame_data(result)

                # Display information.
                cv2.putText(
                    frame,
                    f"Hands detected: {hand_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                
                if recording:

                    if hand_count == 0:
                        cv2.putText(
                            frame,
                            "NO HAND DETECTED - FRAME SKIPPED",
                            (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

                    else:
                        row = [
                            sample_id,
                            LABEL,
                            captured_at,
                            frame_index,
                            hand_count
                        ]

                        row.extend(hand_data["left"])
                        row.extend(hand_data["right"])

                        sample_rows.append(row)
                        frame_index += 1

                        cv2.putText(
                            frame,
                            f"RECORDING: {frame_index}/{FRAMES_PER_SAMPLE}",
                            (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        if frame_index >= FRAMES_PER_SAMPLE:
                            save_sample(sample_rows)

                            sample_rows = []
                            recording = False
                            frame_index = 0

                            print("Sample recording complete.")

                else:
                    cv2.putText(
                        frame,
                        "Press S to record",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )


                cv2.imshow("SignSpeak AI - Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Exiting data collection.")
                    break

                elif key == ord("s") and not recording:
                    sample_id = str(uuid.uuid4())
                    captured_at = datetime.now().isoformat(
                        timespec="seconds"
                    )

                    sample_rows = []
                    frame_index = 0
                    recording = True

                    print("\nRecording started...")
                    print(f"Perform the ISL sign for {LABEL} now.")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


import cv2
import mediapipe as mp
import time
from pathlib import Path

# Project directory and local model path
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Connections between the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_hand_landmarks(frame, landmarks):
    height, width, _ = frame.shape
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    # Draw connections
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    # Draw landmark points
    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)


def main():
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

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam could not be opened.")
        return

    print("Hand tracking started.")
    print("Press Q to quit.")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:
                success, frame = cap.read()

                if not success:
                    print("ERROR: Could not read webcam frame.")
                    break

                # Mirror the webcam image
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(time.monotonic() * 1000)

                # Detect hands
                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                hand_count = len(result.hand_landmarks)

                # Draw detected hand landmarks
                for hand_landmarks in result.hand_landmarks:
                    draw_hand_landmarks(
                        frame,
                        hand_landmarks
                    )

                cv2.putText(
                    frame,
                    f"Hands Detected: {hand_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

                cv2.imshow(
                    "SignSpeak AI - Hand Tracking",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. Hand tracking stopped.")


if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
import time
from pathlib import Path

# Project folder
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def main():

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Webcam not available.")
        return

    print("Coordinate extraction started.")
    print("Show your hand in front of the camera.")
    print("Press Q to quit.\n")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:

                success, frame = camera.read()

                if not success:
                    print("Could not read camera frame.")
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(time.monotonic() * 1000)

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp
                )

                for hand_number, landmarks in enumerate(
                    result.hand_landmarks, start=1
                ):

                    print(f"\n--- HAND {hand_number} ---")

                    for index, point in enumerate(landmarks):

                        print(
                            f"Landmark {index:02d}: "
                            f"X={point.x:.4f}, "
                            f"Y={point.y:.4f}, "
                            f"Z={point.z:.4f}"
                        )

                cv2.imshow(
                    "SignSpeak AI - Coordinate Extraction",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nCamera released. Program stopped.")


if __name__ == "__main__":
    main()
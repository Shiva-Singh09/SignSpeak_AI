
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "signspeak_mlp_best.pth"
LANDMARKER_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# MODEL ARCHITECTURE
# Must match the trained model exactly.
# ---------------------------------------------------------

class SignLanguageMLP(nn.Module):

    def __init__(self, input_size=132, num_classes=26):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    input_size = int(checkpoint["input_features"])
    num_classes = len(checkpoint["class_to_idx"])

    if input_size != 132:
        raise ValueError(
            f"Expected 132 input features, found {input_size}"
        )

    model = SignLanguageMLP(
        input_size=input_size,
        num_classes=num_classes
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    scaler_mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32
    )

    scaler_std = np.asarray(
        checkpoint["scaler_std"],
        dtype=np.float32
    )

    if len(scaler_mean) != 126 or len(scaler_std) != 126:
        raise ValueError("Unexpected scaler dimensions in checkpoint.")

    class_to_idx = checkpoint["class_to_idx"]

    idx_to_class = {
        index: label
        for label, index in class_to_idx.items()
    }

    print("Model loaded successfully.")
    print(f"Device: {DEVICE}")
    print(f"Classes: {num_classes}")

    return model, scaler_mean, scaler_std, idx_to_class


# ---------------------------------------------------------
# FEATURE EXTRACTION
# Must match prepare_realsign_features.py.
# ---------------------------------------------------------

def extract_hand_features(hand_landmarks, handedness):

    coordinates = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand_landmarks
        ],
        dtype=np.float32
    )

    if coordinates.shape != (21, 3):
        raise ValueError(
            f"Expected 21 landmarks, got {coordinates.shape}"
        )

    wrist = coordinates[0].copy()

    relative = coordinates[1:] - wrist

    distances = np.linalg.norm(
        relative,
        axis=1
    )

    scale = float(np.max(distances))

    if scale > 1e-8:
        relative = relative / scale
    else:
        relative = np.zeros_like(relative)

    features = np.concatenate([
        wrist,
        relative.flatten()
    ]).astype(np.float32)

    # MediaPipe handedness category.
    category = handedness[0].category_name

    is_left = int(category == "Left")
    is_right = int(category == "Right")

    return features, 1, is_left, is_right


def extract_frame_features(result):

    # Two hand slots, each containing 63 features.
    hand_features = [
        np.zeros(63, dtype=np.float32),
        np.zeros(63, dtype=np.float32)
    ]

    presence_flags = [0, 0]

    handedness_flags = [
        0, 0, 0, 0
    ]

    if result.hand_landmarks:

        number_of_hands = min(
            len(result.hand_landmarks),
            2
        )

        for hand_index in range(number_of_hands):

            features, present, is_left, is_right = (
                extract_hand_features(
                    result.hand_landmarks[hand_index],
                    result.handedness[hand_index]
                )
            )

            hand_features[hand_index] = features
            presence_flags[hand_index] = present

            if hand_index == 0:
                handedness_flags[0] = is_left
                handedness_flags[1] = is_right
            else:
                handedness_flags[2] = is_left
                handedness_flags[3] = is_right

    combined = np.concatenate([
        hand_features[0],
        hand_features[1],
        np.asarray(
            presence_flags + handedness_flags,
            dtype=np.float32
        )
    ])

    if combined.size != 132:
        raise ValueError(
            f"Expected 132 features, got {combined.size}"
        )

    return combined


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

def predict_sign(
    model,
    features,
    scaler_mean,
    scaler_std,
    idx_to_class
):

    if features.size != 132:
        raise ValueError("Invalid feature vector size.")

    # Standardize only the first 126 coordinate features.
    normalized = features.copy()

    normalized[:126] = (
        normalized[:126] - scaler_mean
    ) / scaler_std

    tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    class_index = int(prediction.item())

    label = idx_to_class[class_index]

    confidence_value = float(
        confidence.item()
    )

    return label, confidence_value


# ---------------------------------------------------------
# DRAW HAND LANDMARKS
# ---------------------------------------------------------

def draw_landmarks(frame, result):

    height, width = frame.shape[:2]

    connections = (
        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    )

    for hand_landmarks in result.hand_landmarks:

        # Draw connections.
        for connection in connections:

            start = hand_landmarks[
                connection.start
            ]

            end = hand_landmarks[
                connection.end
            ]

            start_point = (
                int(start.x * width),
                int(start.y * height)
            )

            end_point = (
                int(end.x * width),
                int(end.y * height)
            )

            cv2.line(
                frame,
                start_point,
                end_point,
                (0, 255, 0),
                2
            )

        # Draw landmark points.
        for landmark in hand_landmarks:

            point = (
                int(landmark.x * width),
                int(landmark.y * height)
            )

            cv2.circle(
                frame,
                point,
                4,
                (0, 100, 255),
                -1
            )


# ---------------------------------------------------------
# MAIN WEBCAM APPLICATION
# ---------------------------------------------------------

def main():

    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {LANDMARKER_PATH}"
        )

    model, scaler_mean, scaler_std, idx_to_class = (
        load_model()
    )

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(LANDMARKER_PATH)
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=(
            mp.tasks.vision.RunningMode.VIDEO
        ),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = (
        mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        landmarker.close()

        raise RuntimeError(
            "Could not open webcam. Check camera permissions."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nSignSpeak AI webcam recognition started.")
    print("Press Q to quit.")

    start_time = time.perf_counter()

    frame_count = 0
    fps = 0.0

    previous_prediction = "Waiting..."
    previous_confidence = 0.0

    # Require a minimum confidence before displaying a letter.
    CONFIDENCE_THRESHOLD = 0.50

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Keep the original camera frame for landmark detection.
            # This helps maintain consistency with training images.
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = (
                time.perf_counter() * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                int(timestamp_ms)
            )

            draw_landmarks(frame, result)

            # Mirror only the displayed camera image.
            frame = cv2.flip(frame, 1)

            if result.hand_landmarks:

                features = extract_frame_features(
                    result
                )

                label, confidence = predict_sign(
                    model,
                    features,
                    scaler_mean,
                    scaler_std,
                    idx_to_class
                )

                if confidence >= CONFIDENCE_THRESHOLD:

                    previous_prediction = label
                    previous_confidence = confidence

                else:

                    previous_prediction = "Uncertain"
                    previous_confidence = confidence

            else:

                previous_prediction = "No hand detected"
                previous_confidence = 0.0

            # FPS calculation.
            frame_count += 1

            elapsed = (
                time.perf_counter() - start_time
            )

            if elapsed >= 1.0:

                fps = frame_count / elapsed

                frame_count = 0
                start_time = time.perf_counter()

            # Prediction panel.
            cv2.rectangle(
                frame,
                (0, 0),
                (640, 115),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                frame,
                f"Prediction: {previous_prediction}",
                (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2
            )

            if previous_prediction not in (
                "No hand detected",
                "Uncertain",
                "Waiting..."
            ):

                confidence_text = (
                    f"Confidence: "
                    f"{previous_confidence * 100:.1f}%"
                )

            elif previous_prediction == "Uncertain":

                confidence_text = (
                    f"Model confidence: "
                    f"{previous_confidence * 100:.1f}%"
                )

            else:

                confidence_text = "Show an alphabet sign"

            cv2.putText(
                frame,
                confidence_text,
                (15, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (15, 103),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (440, 103),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1
            )

            cv2.imshow(
                "SignSpeak AI - Real-Time ISL Recognition",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()

        print("\nSignSpeak AI webcam application closed.")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:

        print(f"\nAPPLICATION ERROR: {error}")

        raise

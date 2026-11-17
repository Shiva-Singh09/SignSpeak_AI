import time
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Selected controlled V3 checkpoint.
MODEL_PATH = (PROJECT_ROOT/ "models"/ "signspeak_mlp_deduplicated_v3_controlled.pth")

LANDMARKER_PATH = (PROJECT_ROOT/ "models"/ "hand_landmarker.task")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# RECOGNITION CONFIGURATION

INPUT_FEATURES = 132
COORDINATE_FEATURES = 126
NUM_CLASSES = 26

# Initial operational threshold.
# This is NOT a statistically calibrated confidence threshold.
CONFIDENCE_THRESHOLD = 0.70

# Temporal probability smoothing.
SMOOTHING_WINDOW = 7

# Require this many consecutive stable predictions.
STABLE_FRAMES = 3

# Webcam settings.
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# MODEL ARCHITECTURE
# Exact V1/V3 architecture.
#
# Linear -> ReLU -> BatchNorm -> Dropout
# Linear -> ReLU -> BatchNorm -> Dropout
# Linear

class SignLanguageMLP(nn.Module):

    def __init__(
        self,
        input_size=INPUT_FEATURES,
        num_classes=NUM_CLASSES
    ):
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

# LOAD V3 MODEL

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(f"V3 checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH,map_location="cpu",weights_only=False
    )

    # CHECKPOINT VALIDATION

    if checkpoint.get("experiment") != "deduplicated_v3_controlled":

        raise ValueError("Unexpected checkpoint. Expected controlled V3 model.")

    input_size = int(checkpoint["input_size"])

    num_classes = int(checkpoint["num_classes"])

    classes = list(checkpoint["classes"])

    expected_classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    if input_size != INPUT_FEATURES:

        raise ValueError(
            f"Expected {INPUT_FEATURES} input features, "
            f"found {input_size}"
        )

    if num_classes != NUM_CLASSES:

        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {num_classes}"
        )

    if classes != expected_classes:

        raise ValueError("Checkpoint class ordering does not match A-Z.")

    # LOAD MODEL

    model = SignLanguageMLP(input_size=input_size,num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"],strict=True)

    model.to(DEVICE)
    model.eval()

    # LOAD V3 NORMALIZATION STATISTICS

    scaler_mean = np.asarray(checkpoint["mean"],dtype=np.float32)

    scaler_std = np.asarray(checkpoint["std"],dtype=np.float32)

    if scaler_mean.shape != (COORDINATE_FEATURES,):

        raise ValueError("Unexpected scaler mean dimensions.")

    if scaler_std.shape != (COORDINATE_FEATURES,):

        raise ValueError("Unexpected scaler standard deviation dimensions.")

    if not np.isfinite(scaler_mean).all():

        raise ValueError("Scaler mean contains non-finite values.")

    if not np.isfinite(scaler_std).all():

        raise ValueError("Scaler standard deviation contains non-finite values.")

    if np.any(scaler_std <= 0):

        raise ValueError("Scaler standard deviations must be positive.")

    idx_to_class = {
        index: label
        for index, label in enumerate(classes)
    }

    print("\nMODEL LOADED SUCCESSFULLY")
    print("-" * 45)

    print("Model: Deduplicated V3 Controlled")
    print("Device:", DEVICE)
    print("Classes:", num_classes)
    print("Input features:", input_size)
    print("Architecture: 132 -> 256 -> 128 -> 26")
    print("Best validation epoch:", checkpoint.get("best_epoch"))
    print("Best validation accuracy:",
        f"{checkpoint.get('best_val_accuracy', 0) * 100:.4f}%"
    )

    return (
        model,
        scaler_mean,
        scaler_std,
        idx_to_class
    )

# FEATURE EXTRACTION
# Must match prepare_deduplicated_features.py.


def extract_hand_features(
    hand_landmarks,
    handedness
):

    coordinates = np.array(
        [
            [landmark.x,landmark.y,landmark.z]
            for landmark in hand_landmarks
        ],
        dtype=np.float32
    )

    if coordinates.shape != (21, 3):
        raise ValueError(f"Expected 21 landmarks, got {coordinates.shape}")

    # Wrist coordinates.
    wrist = coordinates[0].copy()

    # Relative coordinates for remaining 20 landmarks.
    relative = coordinates[1:] - wrist

    # Normalize by maximum 3D distance from wrist.
    distances = np.linalg.norm(relative,axis=1)

    scale = float(np.max(distances))

    if scale > 1e-8:
        relative = relative / scale

    else:

        relative = np.zeros_like(relative)

    features = np.concatenate([wrist,relative.flatten()]).astype(np.float32)

    if features.shape != (63,):

        raise ValueError("Invalid hand feature dimensions.")

    # MediaPipe handedness.
    category = handedness[0].category_name

    is_left = int(category == "Left")

    is_right = int(category == "Right")

    return (
        features,
        1,
        is_left,
        is_right
    )

# EXTRACT FRAME FEATURES

def extract_frame_features(result):

    # Two hand slots, each with 63 coordinates.
    hand_features = [np.zeros(63, dtype=np.float32),np.zeros(63, dtype=np.float32)]

    presence_flags = [0, 0]

    handedness_flags = [0, 0, 0, 0]

    if result.hand_landmarks:

        number_of_hands = min(
            len(result.hand_landmarks),
            2
        )

        for hand_index in range(number_of_hands):
            (
                features,
                present,
                is_left,
                is_right
            ) = extract_hand_features(
                result.hand_landmarks[hand_index],
                result.handedness[hand_index]
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

    if combined.shape != (INPUT_FEATURES,):

        raise ValueError(
            f"Expected {INPUT_FEATURES} features, "
            f"got {combined.size}"
        )

    return combined

# PREDICT CLASS PROBABILITIES

def predict_probabilities(
    model,
    features,
    scaler_mean,
    scaler_std
):

    if features.shape != (INPUT_FEATURES,):

        raise ValueError("Invalid input feature vector.")

    normalized = features.copy()

    # Standardize only the first 126 coordinate features.
    # Presence and handedness flags remain unchanged.

    normalized[:COORDINATE_FEATURES] = (normalized[:COORDINATE_FEATURES]-scaler_mean) / scaler_std

    tensor = torch.from_numpy(normalized).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.softmax(logits,dim=1)

    return probabilities.squeeze(0).cpu().numpy()


# TEMPORAL PREDICTION SMOOTHER

class PredictionSmoother:

    def __init__(
        self,
        window_size=SMOOTHING_WINDOW,
        stable_frames=STABLE_FRAMES,
        confidence_threshold=CONFIDENCE_THRESHOLD
    ):

        self.probability_history = deque(maxlen=window_size)

        self.stable_frames = stable_frames

        self.confidence_threshold = confidence_threshold

        self.previous_candidate = None

        self.candidate_count = 0

        self.current_label = "Waiting..."

        self.current_confidence = 0.0

    def reset(self):

        self.probability_history.clear()

        self.previous_candidate = None

        self.candidate_count = 0

        self.current_label = "Waiting..."

        self.current_confidence = 0.0

    def update(self,probabilities,idx_to_class):

        self.probability_history.append(probabilities.copy())

        # Average probabilities over recent frames.
        averaged_probabilities = np.mean(
            np.stack(list(self.probability_history),axis=0),
            axis=0
        )

        class_index = int(np.argmax(averaged_probabilities))

        confidence = float(averaged_probabilities[class_index])

        candidate = idx_to_class[class_index]

        # CONFIDENCE REJECTION


        if confidence < self.confidence_threshold:

            self.previous_candidate = None

            self.candidate_count = 0

            self.current_label = "Uncertain"

            self.current_confidence = confidence

            return (
                self.current_label,
                self.current_confidence
            )

        # CONSECUTIVE FRAME CONFIRMATION

        if candidate == self.previous_candidate:

            self.candidate_count += 1

        else:

            self.previous_candidate = candidate

            self.candidate_count = 1

        if self.candidate_count >= self.stable_frames:

            self.current_label = candidate

        else:

            self.current_label = "Detecting..."

        self.current_confidence = confidence

        return (
            self.current_label,
            self.current_confidence
        )

# DRAW HAND LANDMARKS

def draw_landmarks(frame, result):

    height, width = frame.shape[:2]

    connections = (mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS)

    for hand_landmarks in result.hand_landmarks:

        # Draw connections.
        for connection in connections:

            start = hand_landmarks[connection.start]

            end = hand_landmarks[connection.end]

            start_point = (
                int(start.x * width),

                int(start.y * height)
            )

            end_point = (
                int(end.x * width),

                int(end.y * height)
            )

            cv2.line(frame,start_point,end_point,(0, 255, 0),2)

        # Draw landmark points.
        for landmark in hand_landmarks:
            point = (
                int(landmark.x * width),

                int(landmark.y * height)
            )

            cv2.circle(frame,point,4,(0, 100, 255),-1)

# DRAW INFORMATION PANEL

def draw_information_panel(frame,prediction,confidence,fps):

    height, width = frame.shape[:2]

    panel_height = 125

    cv2.rectangle(frame,(0, 0),(width, panel_height),(25, 25, 25),-1)

    # Prediction.
    cv2.putText(frame,f"Prediction: {prediction}",(15, 38),cv2.FONT_HERSHEY_SIMPLEX,0.80,(0, 255, 0),2)

    # Confidence.
    if prediction in ("No hand detected","Waiting...","Detecting..."):

        confidence_text = "Show and hold an alphabet sign"

    elif prediction == "Uncertain":

        confidence_text = (f"Model confidence: {confidence * 100:.1f}%")

    else:

        confidence_text = (f"Confidence: {confidence * 100:.1f}%")

    cv2.putText(frame,confidence_text,(15, 73),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255, 255, 255),2)

    # FPS.
    cv2.putText(frame,f"FPS: {fps:.1f}",(15, 105),cv2.FONT_HERSHEY_SIMPLEX,0.50,(200, 200, 200),1)

    cv2.putText(frame,"Press Q to quit",(width - 155, 105),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200, 200, 200),1)

# MAIN WEBCAM APPLICATION

def main():

    # CHECK MEDIAPIPE MODEL

    if not LANDMARKER_PATH.exists():

        raise FileNotFoundError(f"MediaPipe model not found: {LANDMARKER_PATH}")

    # LOAD CLASSIFIER

    (
        model,
        scaler_mean,
        scaler_std,
        idx_to_class
    ) = load_model()

    # CREATE MEDIAPIPE LANDMARKER

    base_options = mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH))

    options = mp.tasks.vision.HandLandmarkerOptions(

        base_options=base_options,

        running_mode=(mp.tasks.vision.RunningMode.VIDEO),

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

    # OPEN WEBCAM

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():

        landmarker.close()

        raise RuntimeError("Could not open webcam. Check camera permissions.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH,CAMERA_WIDTH)

    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,CAMERA_HEIGHT)

    # INITIALIZE RECOGNITION STATE

    smoother = PredictionSmoother()

    fps = 0.0

    fps_frame_count = 0

    fps_start_time = time.perf_counter()

    # MediaPipe VIDEO mode requires monotonically increasing timestamps.
    previous_timestamp_ms = -1

    print("\n" + "=" * 55)

    print("SIGNSPEAK AI - REAL-TIME ISL RECOGNITION")

    print("=" * 55)

    print("Model: V3 Controlled")

    print("Device:", DEVICE)

    print("Confidence threshold:", CONFIDENCE_THRESHOLD)

    print("Temporal smoothing window:", SMOOTHING_WINDOW)

    print("Stable frames required:", STABLE_FRAMES)

    print("\nPress Q to quit.")

    try:

        while True:

            # CAPTURE FRAME

            success, frame = camera.read()

            if not success:

                print("Could not read webcam frame.")

                break

            # PREPARE FRAME FOR MEDIAPIPE

            rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)

            timestamp_ms = int(time.perf_counter() * 1000)

            timestamp_ms = max(timestamp_ms,previous_timestamp_ms + 1)

            previous_timestamp_ms = timestamp_ms

            # HAND DETECTION

            result = landmarker.detect_for_video(mp_image,timestamp_ms)

            draw_landmarks(frame,result)

            # NO-HAND REJECTION

            if not result.hand_landmarks:

                # Do not run the alphabet classifier.
                # Reset temporal history to prevent stale predictions.

                smoother.reset()

                prediction = "No hand detected"

                confidence = 0.0

            else:

                # FEATURE EXTRACTION

                features = extract_frame_features(result)

                hand_0_present = features[126]

                hand_1_present = features[127]

                if (hand_0_present == 0 and hand_1_present == 0):

                    smoother.reset()

                    prediction = "No hand detected"

                    confidence = 0.0

                else:

                    # CLASSIFIER INFERENCE

                    probabilities = predict_probabilities(model,features,scaler_mean,scaler_std)

                    # TEMPORAL SMOOTHING


                    prediction, confidence = smoother.update(probabilities,idx_to_class)

            # MIRROR DISPLAY

            # Detection and feature extraction use the original frame.
            # Only the displayed image is mirrored.

            frame = cv2.flip(frame,1)

            # FPS CALCULATION

            fps_frame_count += 1

            elapsed = (time.perf_counter() - fps_start_time)

            if elapsed >= 1.0:

                fps = (fps_frame_count / elapsed)

                fps_frame_count = 0

                fps_start_time = time.perf_counter()

            # DISPLAY INFORMATION

            draw_information_panel(frame,prediction,confidence,fps)

            cv2.imshow("SignSpeak AI - Real-Time ISL Recognition",frame)

            # KEYBOARD INPUT

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

    finally:

        camera.release()

        landmarker.close()

        cv2.destroyAllWindows()

        print("\nSignSpeak AI webcam application closed.")


# ENTRY POINT

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(f"\nAPPLICATION ERROR: {error}")
        raise
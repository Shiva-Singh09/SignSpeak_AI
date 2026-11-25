import sys
import time
from pathlib import Path

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition import realtime_recognition as recognition

CHART_PATH = PROJECT_ROOT / "app" / "assets" / "isl_alphabet_reference.png"
LANDMARKER_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SignSpeak AI | ISL Recognition",
    page_icon="🤟",
    layout="wide",
)

st.title("SignSpeak AI")
st.caption("Real-Time Indian Sign Language Alphabet Recognition")

st.markdown(
    """
    Show an ISL alphabet sign to your webcam and hold it steady.
    The system will detect your hand landmarks and predict the alphabet.
    """
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading SignSpeak AI model...")
def get_model_bundle():
    return recognition.load_model()


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class SignSpeakVideoProcessor(VideoProcessorBase):

    def __init__(self, model_bundle):
        self.model, self.scaler_mean, self.scaler_std, self.idx_to_class = (
            model_bundle
        )

        self.smoother = recognition.PredictionSmoother()
        self.previous_timestamp_ms = -1

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(LANDMARKER_PATH)
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )

        timestamp_ms = int(time.perf_counter() * 1000)

        timestamp_ms = max(
            timestamp_ms,
            self.previous_timestamp_ms + 1,
        )

        self.previous_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        recognition.draw_landmarks(image, result)

        if not result.hand_landmarks:

            self.smoother.reset()
            prediction = "No hand detected"
            confidence = 0.0

        else:

            features = recognition.extract_frame_features(result)

            if features[126] == 0 and features[127] == 0:

                self.smoother.reset()
                prediction = "No hand detected"
                confidence = 0.0

            else:

                probabilities = recognition.predict_probabilities(
                    self.model,
                    features,
                    self.scaler_mean,
                    self.scaler_std,
                )

                prediction, confidence = self.smoother.update(
                    probabilities,
                    self.idx_to_class,
                )

        # Mirror the displayed webcam feed.
        image = cv2.flip(image, 1)

        # Prediction panel.
        cv2.rectangle(
            image,
            (0, 0),
            (image.shape[1], 115),
            (25, 25, 25),
            -1,
        )

        cv2.putText(
            image,
            f"Prediction: {prediction}",
            (15, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
        )

        if prediction in (
            "No hand detected",
            "Waiting...",
            "Detecting...",
        ):
            confidence_text = "Show and hold an alphabet sign"

        else:
            confidence_text = (
                f"Confidence: {confidence * 100:.1f}%"
            )

        cv2.putText(
            image,
            confidence_text,
            (15, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )

    def __del__(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


# ============================================================
# MAIN INTERFACE
# ============================================================

if not CHART_PATH.exists():
    st.error(f"ISL reference chart not found: {CHART_PATH}")
    st.stop()

if not LANDMARKER_PATH.exists():
    st.error(f"MediaPipe model not found: {LANDMARKER_PATH}")
    st.stop()

model_bundle = get_model_bundle()

left_column, right_column = st.columns(
    [1, 1.25],
    gap="large",
)

with left_column:

    st.subheader("ISL Alphabet Reference")

    st.image(
        str(CHART_PATH),
        width="stretch",
    )

    st.caption("Reference: Official ISL alphabet PDF")

with right_column:

    st.subheader("Live Recognition")

    st.info(
        "Allow camera permission in your browser. "
        "Hold an alphabet sign steadily in front of the camera."
    )

    webrtc_streamer(
        key="signspeak-live-recognition",
        video_processor_factory=lambda: SignSpeakVideoProcessor(
            model_bundle
        ),
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {
                    "urls": ["stun:stun.l.google.com:19302"]
                }
            ]
        },
    )

st.divider()

st.caption(
    "SignSpeak AI | ISL A-Z | Deduplicated V3 Controlled Model"
)

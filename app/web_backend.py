import base64
import time
from contextlib import asynccontextmanager
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from src.recognition import realtime_recognition as recognition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_IMAGE = PROJECT_ROOT / "app" / "assets" / "isl_alphabet_reference.png"

model_bundle = None


def initialize_model():
    global model_bundle

    if model_bundle is None:
        model_bundle = recognition.load_model()

    return model_bundle


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SignSpeak AI - ISL Recognition</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    padding: 24px;
    background: #101827;
    color: #f8fafc;
    font-family: Arial, sans-serif;
}
h1 { margin-bottom: 6px; }
.subtitle { color: #aab7ca; margin-bottom: 24px; }
.layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px;
}
.panel {
    background: #1c293b;
    padding: 18px;
    border-radius: 12px;
}
.panel h2 { margin-top: 0; font-size: 20px; }
.reference {
    width: 100%;
    height: auto;
    border-radius: 8px;
    background: white;
}
#output {
    width: 100%;
    background: #080d16;
    border-radius: 8px;
    min-height: 240px;
    object-fit: contain;
}
.controls { margin: 14px 0; }
button {
    border: 0;
    border-radius: 7px;
    padding: 11px 18px;
    background: #2563eb;
    color: white;
    font-size: 15px;
    cursor: pointer;
}
button:disabled { opacity: .5; cursor: not-allowed; }
#prediction { font-size: 28px; font-weight: bold; margin-top: 14px; }
#confidence, #status { color: #b9c7d9; margin-top: 8px; }
@media (max-width: 800px) {
    body { padding: 12px; }
    .layout { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<h1>SignSpeak AI</h1>
<div class="subtitle">Indian Sign Language - Alphabet Recognition (A-Z)</div>

<div class="layout">
    <section class="panel">
        <h2>ISL Alphabet Reference</h2>
        <img class="reference" src="/reference" alt="ISL alphabet reference chart">
    </section>

    <section class="panel">
        <h2>Live Recognition</h2>
        <video id="video" autoplay playsinline muted style="display:none"></video>
        <canvas id="capture" width="640" height="480" style="display:none"></canvas>
        <img id="output" alt="Live annotated webcam feed">
        <div class="controls">
            <button id="start">Start Camera</button>
            <button id="stop" disabled>Stop Camera</button>
        </div>
        <div id="prediction">Waiting...</div>
        <div id="confidence">Confidence: --</div>
        <div id="status">Camera is stopped.</div>
    </section>
</div>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("capture");
const ctx = canvas.getContext("2d");
const output = document.getElementById("output");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const statusText = document.getElementById("status");

let stream = null;
let socket = null;
let busy = false;
let active = false;

async function startCamera() {
    try {
        statusText.textContent = "Requesting camera permission...";
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        socket = new WebSocket(
            (location.protocol === "https:" ? "wss://" : "ws://")
            + location.host + "/ws"
        );

        socket.onopen = () => {
            active = true;
            startButton.disabled = true;
            stopButton.disabled = false;
            statusText.textContent = "Camera connected. Recognizing...";
            sendFrame();
        };

        socket.onmessage = (event) => {
            busy = false;
            const result = JSON.parse(event.data);

            if (result.frame) {
                output.src = "data:image/jpeg;base64," + result.frame;
            }

            prediction.textContent = result.prediction;
            confidence.textContent =
                "Confidence: " + (result.confidence * 100).toFixed(2) + "%";

            if (active) {
                sendFrame();
            }
        };

        socket.onerror = () => {
            statusText.textContent = "WebSocket connection error.";
        };

        socket.onclose = () => {
            if (active) stopCamera();
        };

    } catch (error) {
        statusText.textContent = "Camera error: " + error.message;
        stopCamera();
    }
}

function sendFrame() {
    if (!active || busy || !socket || socket.readyState !== WebSocket.OPEN) {
        return;
    }

    if (!video.videoWidth || !video.videoHeight) {
        requestAnimationFrame(sendFrame);
        return;
    }

    busy = true;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        if (!active || !blob || socket.readyState !== WebSocket.OPEN) {
            busy = false;
            return;
        }
        socket.send(await blob.arrayBuffer());
    }, "image/jpeg", 0.75);
}

function stopCamera() {
    active = false;
    busy = false;

    if (socket) {
        socket.close();
        socket = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    startButton.disabled = false;
    stopButton.disabled = true;
    statusText.textContent = "Camera is stopped.";
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("beforeunload", stopCamera);
</script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app):
    initialize_model()
    yield


app = FastAPI(title="SignSpeak AI", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML


@app.get("/reference")
async def reference():
    if not REFERENCE_IMAGE.exists():
        return HTMLResponse("ISL reference image not found.", status_code=404)

    return FileResponse(REFERENCE_IMAGE)


@app.websocket("/ws")
async def recognition_socket(websocket: WebSocket):
    await websocket.accept()

    model, scaler_mean, scaler_std, idx_to_class = initialize_model()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(recognition.LANDMARKER_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    smoother = recognition.PredictionSmoother()
    previous_timestamp_ms = -1

    try:
        while True:
            image_bytes = await websocket.receive_bytes()

            frame_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = max(
                int(time.perf_counter() * 1000),
                previous_timestamp_ms + 1,
            )
            previous_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.hand_landmarks:
                smoother.reset()
                label = "No hand detected"
                confidence_value = 0.0
            else:
                features = recognition.extract_frame_features(result)

                if features[126] == 0 and features[127] == 0:
                    smoother.reset()
                    label = "No hand detected"
                    confidence_value = 0.0
                else:
                    probabilities = recognition.predict_probabilities(
                        model,
                        features,
                        scaler_mean,
                        scaler_std,
                    )
                    label, confidence_value = smoother.update(
                        probabilities,
                        idx_to_class,
                    )

            recognition.draw_landmarks(frame, result)
            frame = cv2.flip(frame, 1)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 76), (25, 35, 52), -1)
            cv2.putText(
                frame, str(label), (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2,
            )
            cv2.putText(
                frame, f"Confidence: {confidence_value * 100:.1f}%",
                (16, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )

            success, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

            if not success:
                continue

            frame_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")

            await websocket.send_json({
                "frame": frame_base64,
                "prediction": str(label),
                "confidence": float(confidence_value),
            })

    except WebSocketDisconnect:
        pass
    finally:
        landmarker.close()


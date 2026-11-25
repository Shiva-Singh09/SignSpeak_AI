# SignSpeak AI - Setup Guide

## 1. Prerequisites

### Python

- Python 3.12.7 is the project-tested version.
- Python 3.11 or later may work, but compatibility should be verified.
- Python 3.10 and earlier are not covered by the current setup.

### Hardware and GPU

The project-tested configuration includes:

- NVIDIA GeForce RTX 4050 Laptop GPU.
- PyTorch 2.14.1 with CUDA 13.0 support.
- Successful CUDA availability and tensor execution tests.

A compatible NVIDIA driver is required for GPU execution. CPU execution may also be used, although performance can differ.

### Operating System

- Windows 10/11 (64-bit): project-tested environment.
- Other operating systems may require platform-specific setup adjustments.

## 2. Project-Local Virtual Environment

Run the following commands from the project root directory.

### Create the environment

    py -3.12 -m venv .venv

Skip this step if the existing project virtual environment is already configured and working.

### Activate the environment in PowerShell

    .\.venv\Scripts\Activate.ps1

### Verify Python

    python --version
    python -c "import sys; print(sys.executable)"

The executable should point to the project's `.venv` directory.

Keep package installations inside this virtual environment.

## 3. Dependency Installation

The following versions were reported in the project-tested environment:

- PyTorch: 2.14.1+cu130
- NumPy: 2.5.4
- MediaPipe: 0.10.35
- OpenCV: 5.0.0

For a fresh environment, activate `.venv` before installing packages.

### Install PyTorch with CUDA support

    python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130

### Install supporting packages

    python -m pip install numpy==2.5.4
    python -m pip install mediapipe==0.10.35
    python -m pip install opencv-python==5.0.0
    python -m pip install onnxruntime pillow

Do not reinstall packages unnecessarily in an already working project environment.

## 4. Verify Dependencies

Run these commands inside the activated virtual environment.

### Check PyTorch and CUDA

    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

### Check MediaPipe

    python -c "import mediapipe as mp; print('MediaPipe:', mp.__version__)"

### Check OpenCV

    python -c "import cv2; print('OpenCV:', cv2.__version__)"

### Check NumPy

    python -c "import numpy as np; print('NumPy:', np.__version__)"

### Check ONNX Runtime

    python -c "import onnxruntime as ort; print('ONNX Runtime:', ort.__version__)"

## 5. Model and Dataset Requirements

### Model files

The project stores model-related files in:

    models/

Important file categories include:

- MediaPipe Hand Landmarker model (`hand_landmarker.task`).
- Trained PyTorch model checkpoints (`.pth`).
- Exported ONNX model (`.onnx`).

The exact checkpoint or exported model required depends on the script being executed. Check the relevant script's configured model path before running it.

### Hand Landmarker asset

The MediaPipe asset used by the hand-tracking pipeline is:

    models/hand_landmarker.task

Ensure this file exists before running scripts that use MediaPipe Hand Landmarker.

### Dataset locations

The main dataset directories are:

- `dataset/raw/` - Raw collected data.
- `dataset/processed/` - Processed landmarks and features.
- `dataset/labels/` - Dataset labels and related information.
- `dataset/external/` - External datasets, including RealSign.

Avoid modifying the original dataset when performing experiments. Keep generated files in the appropriate project directories.

## 6. Running the Project

Run commands from the project root with `.venv` activated.

### Real-Time Recognition

    python -m src.recognition.realtime_recognition

### Hand Tracking Demonstration

    python src/hand_tracking/hand_tracking_demo.py

### Model Training

Basic MLP training:

    python src/training/train_deduplicated_mlp.py

Controlled V3 training:

    python src/training/train_deduplicated_mlp_v3_controlled.py

### Model Evaluation

    python src/evaluation/evaluate_mlp.py

Other dataset-analysis and model-comparison scripts are available under:

    src/evaluation/

Check each script's inputs and output paths before running it.

## 7. Troubleshooting

### CUDA is unavailable

- Confirm that PyTorch is installed in the active `.venv`.
- Check `torch.cuda.is_available()`.
- Verify NVIDIA driver compatibility.
- Confirm that the installed PyTorch build supports the intended CUDA configuration.

### MediaPipe import error

Run:

    python -m pip show mediapipe

Confirm that the active Python executable belongs to `.venv`.

### OpenCV import error

Run:

    python -m pip show opencv-python

Check that the project virtual environment is active.

### ONNX Runtime import error

Run:

    python -m pip show onnxruntime

Install or repair the package inside `.venv` if it is missing.

### Camera cannot be opened

- Close other programs that may be using the camera.
- Check operating system camera access settings.
- Verify the camera index used by the recognition script.
- Check that the camera is connected and functioning.

### Hand not detected

- Ensure sufficient lighting.
- Keep the hand visible within the camera frame.
- Avoid excessive hand occlusion.
- Check the MediaPipe detection configuration.

### Model file not found

- Confirm that the expected file exists in `models/`.
- Check the model path configured in the script.
- Run commands from the project root directory.

## 8. Setting Up on Another Computer

1. Copy the project folder to the target computer.
2. Install a compatible Python version.
3. Open the project root directory in PowerShell.
4. Create a project-local virtual environment:

       py -3.12 -m venv .venv

5. Activate it:

       .\.venv\Scripts\Activate.ps1

6. Install the required packages inside `.venv`.
7. Verify the required model assets and dataset files.
8. Run the dependency verification commands.
9. Start with the hand-tracking demonstration or the recognition script.

## 9. Project Structure Overview

    SignSpeak_AI/
    |
    +-- .venv/
    |   +-- Project-local Python environment
    |
    +-- dataset/
    |   +-- raw/
    |   +-- processed/
    |   +-- labels/
    |   +-- external/
    |
    +-- models/
    |   +-- Hand Landmarker model
    |   +-- PyTorch checkpoints
    |   +-- ONNX exports
    |
    +-- src/
    |   +-- data_collection/
    |   +-- hand_tracking/
    |   +-- preprocessing/
    |   +-- training/
    |   +-- evaluation/
    |   +-- recognition/
    |
    +-- outputs/
    |   +-- graphs/
    |   +-- reports/
    |
    +-- notebooks/
    +-- cache/
    +-- temp/
    +-- README.md
    +-- SETUP.md
    +-- DATASET.md
    +-- TECHNICAL_DETAILS.md
    +-- MODEL_EVALUATION.md
    +-- DEVELOPMENT_JOURNAL.md

## 10. Environment Isolation

The project is intended to keep its Python environment, packages, datasets, models, temporary files, and generated outputs within the project directory.

- Activate `.venv` before installing or running project dependencies.
- Avoid installing project packages into global Python.
- Use project-local directories for temporary files and caches.
- Do not change system-wide Python configuration for project setup.
- Keep generated experiments and reports inside `outputs/` or their relevant project directories.

---

**SignSpeak AI - Setup Guide**

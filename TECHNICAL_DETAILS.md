# SignSpeak AI - Technical Details

## 1. Project Overview

SignSpeak AI is a machine learning project designed to recognize Indian Sign Language (ISL) alphabet signs.

The first target is static alphabet recognition (A-Z). The project uses hand landmark detection and a machine learning classifier.

The main workflow converts detected hand landmarks into numerical features and predicts the corresponding alphabet class.

## 2. Main Technologies

- Python: Main programming language.
- PyTorch: Model development and training.
- MediaPipe: Hand landmark detection.
- OpenCV: Camera access and image processing.
- NumPy: Numerical operations.
- ONNX: Portable model representation.
- ONNX Runtime: ONNX model inference and validation.

The project uses a local Python virtual environment.

## 3. Project Architecture

The main recognition workflow is:

1. Capture an image from the camera.
2. Convert the image into the required color format.
3. Detect hands using MediaPipe.
4. Extract 21 landmarks from each detected hand.
5. Convert landmarks into numerical features.
6. Normalize and standardize the features as required.
7. Pass the feature vector to the trained MLP model.
8. Obtain predictions for the 26 alphabet classes.

## 4. Hand Landmark Extraction

MediaPipe Hand Landmarker detects 21 landmarks for each hand.

Each landmark contains three coordinates:

- X: Horizontal position.
- Y: Vertical position.
- Z: Depth information.

Therefore:

21 landmarks x 3 coordinates = 63 features per hand.

For two hand slots:

63 x 2 = 126 coordinate features.

## 5. Feature Engineering

The model uses 132 input features.

### Coordinate Features

The first 126 features represent two hand slots:

- First slot: Left hand.
- Second slot: Right hand.

Each hand slot contains 63 coordinate values.

The wrist landmark (landmark 0) is retained in its original coordinate form.

For landmarks 1 to 20, wrist-relative coordinates are calculated.

The relative coordinates are normalized using the maximum Euclidean distance from the wrist among the 20 relative landmarks.

If the scale is too small (less than or equal to 1e-8), the relative coordinates are set to zero to avoid division problems.

If a hand is missing, its 63 coordinate values are filled with zeros.

### Hand Presence and Handedness Flags

Six additional features describe hand presence and handedness:

1. hand_0_present
2. hand_1_present
3. hand_0_is_left
4. hand_0_is_right
5. hand_1_is_left
6. hand_1_is_right

Total:

126 coordinate features + 6 flags = 132 features.

The hand slots are assigned using handedness information, not simply the order in which MediaPipe returns detected hands.

## 6. Feature Standardization

The model standardizes the first 126 coordinate features using training dataset statistics.

Formula:

standardized_value = (value - mean) / standard_deviation

The six hand-related flags are not standardized.

If a standard deviation is smaller than 1e-8, it is replaced with 1 to avoid division by an extremely small value.

The mean and standard deviation are stored in the trained model checkpoint.

The ONNX export incorporates these scaling values into the exported model.

Therefore, an inference pipeline using this ONNX model should provide the raw 132-feature vector and should not apply the same standardization a second time.

## 7. Machine Learning Model

The project uses a Multi-Layer Perceptron (MLP).

The V3 controlled model architecture is:

- Input layer: 132 features
- Hidden layer 1: 256 neurons
- ReLU activation
- Batch Normalization
- Dropout: 0.30
- Hidden layer 2: 128 neurons
- ReLU activation
- Batch Normalization
- Dropout: 0.20
- Output layer: 26 classes

The output represents the 26 alphabet classes from A to Z.

The model produces raw logits. Softmax can be applied to convert logits into class probabilities.

## 8. Model Experiments

Different model versions were developed:

- V1: Initial deduplicated MLP experiment.
- V2: Hand-present model experiment.
- V3: Controlled training experiment.
- V4: Gradient diagnostic experiment.

The project also contains earlier MLP checkpoints.

Each experiment has its own checkpoint and evaluation artifacts. Their results should not be treated as interchangeable.

## 9. Dataset and Preprocessing

The project contains self-collected alphabet landmark data and an external RealSign dataset.

The preprocessing and evaluation workflow includes:

- Landmark extraction.
- Landmark cleaning.
- Handedness inspection.
- No-hand sample filtering.
- Feature preparation.
- Duplicate detection.
- Near-duplicate analysis.
- Dataset split verification.
- Dataset quality checks.

Exact dataset counts and split details should be taken from the corresponding dataset summaries and audit reports.

## 10. ONNX Export

The V3 controlled PyTorch model is exported to ONNX.

PyTorch checkpoint:

models/signspeak_mlp_deduplicated_v3_controlled.pth

The exported ONNX model is stored in the models directory.

Export configuration:

- Input name: features
- Output name: logits
- Input size: 132
- Output classes: 26
- ONNX opset: 17
- Dynamic batch dimension: Supported

The export workflow checks the model configuration, scaler values and ONNX structure.

## 11. ONNX Validation

The project includes a validation workflow that compares PyTorch and ONNX predictions.

The recorded comparison uses 512 synthetic feature samples and tests different hand-presence combinations.

The comparison checks:

- Numerical output differences.
- Prediction agreement.
- Numerical tolerance.

A previous parity check reported 100% prediction agreement, with a maximum absolute output difference of approximately 3.81e-06.

This test checks model conversion consistency. It does not independently establish real-world recognition accuracy or verify the complete camera-to-prediction pipeline.

## 12. Important Limitations

- The first target is static alphabet recognition, not complete sign language sentence translation.
- Model performance depends on dataset quality and signer variation.
- Testing with more signers and different lighting conditions is important.
- Validation accuracy does not automatically represent real-world accuracy.
- Synthetic ONNX testing does not replace testing with real input data.
- The complete real-time recognition pipeline requires further testing under varied conditions.

## 13. Main Project Directories

- dataset/: Raw and processed datasets.
- models/: Trained models and hand landmark model.
- src/data_collection/: Data collection scripts.
- src/hand_tracking/: Hand tracking scripts.
- src/preprocessing/: Data preprocessing scripts.
- src/training/: Model training and export scripts.
- src/evaluation/: Evaluation and audit scripts.
- src/recognition/: Recognition scripts.
- outputs/: Reports, graphs and evaluation results.

---

Document status: Technical details are based on the available project information and previously recorded experiment results. Dataset counts and experiment-specific metrics should be checked against their corresponding reports.

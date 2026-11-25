# SignSpeak AI

## 1. Project Introduction

SignSpeak AI is an Indian Sign Language (ISL) alphabet-recognition project. It focuses on classifying static hand gestures representing the English alphabet from A to Z.

The project uses hand landmark detection, feature extraction, preprocessing, and machine learning to predict an alphabet class from a hand gesture.

## 2. Problem Statement

Recognizing hand gestures accurately is an important part of developing assistive sign-language technology.

Building a reliable recognition system involves several challenges:

- Detecting hands under different visual conditions.
- Extracting consistent hand landmark coordinates.
- Handling missing-hand frames.
- Maintaining correct left-hand and right-hand feature ordering.
- Preparing input features consistently with the training process.
- Distinguishing visually similar alphabet gestures.
- Evaluating performance on data that was not used during training.

## 3. Project Objectives

1. **ISL Alphabet Recognition:** Classify static ISL alphabet gestures from A to Z.
2. **Hand Landmark Detection:** Detect hand landmarks using MediaPipe.
3. **Feature Engineering:** Convert detected landmarks into a consistent 132-feature representation.
4. **Machine Learning:** Train and evaluate an MLP classifier for 26 alphabet classes.
5. **Data Quality:** Analyse missing-hand samples, duplicate data, and dataset consistency.
6. **Model Evaluation:** Measure accuracy, precision, recall, F1-score, and class-wise errors.

## 4. What SignSpeak AI Does

The recognition pipeline is designed to:

- Process camera frames using the Python-based computer vision pipeline.
- Detect one or two hands using MediaPipe Hand Landmarker.
- Extract hand landmark coordinates.
- Prepare the expected 132 input features.
- Apply model-compatible preprocessing.
- Run the trained classification model.
- Predict one alphabet class from A to Z.

## 5. Supported Alphabet Classes

The current classification task contains 26 static alphabet classes:

A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z.

## 6. System Architecture

The main recognition pipeline is:

```text
Camera Frame
     |
     v
OpenCV Frame Processing
     |
     v
MediaPipe Hand Landmarker
     |
     v
Hand Landmark Coordinates
     |
     v
132-Feature Extraction
     |
     v
Feature Preprocessing
     |
     v
MLP Model Inference
     |
     v
Predicted ISL Alphabet (A-Z)
```

### Main Components

- **OpenCV:** Image and video frame processing.
- **MediaPipe Hand Landmarker:** Detects 21 landmarks per hand.
- **Feature Extraction:** Creates the model's 132 input features.
- **Preprocessing:** Applies the feature preparation and standardization used by the trained model.
- **MLP Classifier:** Predicts one of the 26 alphabet classes.
- **ONNX:** Provides an exported model format for inference and numerical consistency experiments.

## 7. Dataset Overview

The project contains self-collected ISL alphabet data and an external RealSign dataset.

Dataset processing includes:

- Alphabet labels from A to Z.
- Hand landmark extraction.
- Data cleaning and validation.
- Missing-hand sample analysis.
- Duplicate and near-duplicate analysis.
- Training, validation, and test data preparation.

The self-collected dataset initially showed a high proportion of frames without detected hands in several samples. Dataset quality and signer diversity therefore remain important areas for improvement.

See `DATASET.md` for more information.

## 8. Model Overview

### Architecture

The controlled V3 MLP architecture is:

```text
Input:       132 features
Hidden 1:    256 neurons
Hidden 2:    128 neurons
Output:       26 classes
```

Architecture: `132 → 256 → 128 → 26`

The network uses ReLU activation, Batch Normalization, and Dropout.

### Input Features

The input contains 132 float32 features:

- 126 coordinate features: 63 for each hand.
- 6 hand-presence and handedness flags.

The feature ordering must remain consistent between data preparation, training, evaluation, and inference.

### Output

The model produces 26 class logits corresponding to A-Z. Softmax can convert these scores into class probabilities.

### Frameworks and Formats

- PyTorch is used for model training and evaluation.
- ONNX is used as an exported model format.
- The V3 ONNX export embeds coordinate standardization. Its expected input is the raw 132-feature vector, so the same standardization must not be applied twice.

## 9. Results Summary

The following results come from separate evaluation runs and must not be treated as directly interchangeable.

### V3 Controlled Validation

- Validation samples: 2,284.
- Accuracy: approximately 99.8249%.
- Macro F1-score: approximately 99.8325%.

This evaluation used a hand-present validation dataset.

### Separate Test Evaluation

For `signspeak_mlp_best.pth`:

- Test samples: 4,630.
- Accuracy: approximately 92.8942%.
- Macro precision: approximately 93.8424%.
- Macro recall: approximately 92.4898%.
- Macro F1-score: approximately 92.4942%.

This is a different checkpoint and evaluation run from the V3 validation result.

### ONNX Numerical Comparison

An earlier comparison reported:

- 512 synthetic input samples.
- 100% prediction agreement between PyTorch and ONNX.
- Maximum absolute difference: approximately 0.0000038147.

This checks numerical consistency for the tested synthetic inputs. It does not establish real-world recognition accuracy.

See `MODEL_EVALUATION.md` for detailed results.

## 10. Important Implementation Details

1. **Consistent feature ordering:** The 132-feature input layout must match the training pipeline.
2. **Handedness handling:** Left-hand and right-hand features must occupy their expected positions.
3. **Missing-hand handling:** Frames without detected hands require correct handling.
4. **Coordinate normalization:** Very small coordinate scales must be handled safely.
5. **Standardization:** Coordinate features use training-derived mean and standard deviation.
6. **Evaluation discipline:** Validation and test results must be reported separately.
7. **Duplicate control:** Exact and near-duplicate samples should be checked to reduce data leakage.

## 11. Current Limitations

- The classifier recognizes static alphabet classes only.
- It does not currently recognize complete words or sentences.
- Performance depends on dataset quality and signer variation.
- Some alphabet classes may be visually confusing.
- More testing with unseen signers is needed.
- Synthetic ONNX comparison does not replace evaluation using real extracted landmarks.
- Real-world recognition performance needs further verification.

## 12. Future Scope

- Collect more samples from different signers.
- Improve data quality and hand-detection consistency.
- Investigate confusing alphabet classes.
- Improve generalization under different lighting and backgrounds.
- Extend recognition to dynamic signs.
- Explore word- and sentence-level recognition.
- Improve model efficiency and prediction stability.
- Expand evaluation using independent datasets and unseen signers.

## 13. Quick Start

1. Use the project-tested Python 3.12.7 environment.
2. Open the project root directory in PowerShell.
3. Activate the project-local virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Check `SETUP.md` for environment and dependency details.
5. Run the real-time recognition script:

   ```powershell
   python -m src.recognition.realtime_recognition
   ```

6. Use the evaluation scripts under `src/evaluation` to inspect model performance and dataset quality.

## 14. Project Documentation

- `SETUP.md` — Environment setup and dependencies.
- `DATASET.md` — Dataset structure and preparation.
- `TECHNICAL_DETAILS.md` — Technical pipeline and model details.
- `MODEL_EVALUATION.md` — Model evaluation and comparison results.
- `DEVELOPMENT_JOURNAL.md` — Development history.
- `outputs/project_audit/` — Project inventory and review documents.

## License

This project is intended for educational and accessibility-related development. Specific license terms should be determined separately.

---

**SignSpeak AI — Indian Sign Language Alphabet Recognition (A-Z).**

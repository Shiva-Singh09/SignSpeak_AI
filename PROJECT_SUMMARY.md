# SignSpeak AI - Project Summary

## Project Overview
SignSpeak AI is a machine learning project designed to recognize Indian Sign Language (ISL) alphabet signs. The first target is static alphabet recognition (A-Z).

## Project Structure
- `dataset/`: Self-collected ISL alphabet dataset and external RealSign dataset
- `models/`: Trained MLP models and hand landmark model
- `src/data_collection/`: Hand landmark collection scripts (collect_landmarks.py)
- `src/preprocessing/`: Data preprocessing pipelines
- `src/training/`: Model training scripts (train_deduplicated_mlp.py and variants)
- `src/recognition/`: Real-time sign recognition scripts (realtime_recognition.py)
- `src/evaluation/`: Model evaluation scripts (evaluate_model.py)
- `outputs/`: Evaluation reports, graphs, and results

## Key Features
1. **Hand Landmark Detection**: MediaPipe Hands API for detecting 21 hand landmarks
2. **Feature Extraction**: 132 features (63 per hand × 2 hands, plus 6 hand presence/handedness flags)
3. **Model Architecture**: Multi-Layer Percepton (MLP) with deduplicated training
4. **ONNX Export**: Models exported to ONNX format for cross-platform inference
5. **Train/Val/Test Splits**: 70/15/15 data split for reliable evaluation

## Workflow
1. **Data Collection**: Capture hand signs via webcam using MediaPipe landmark extraction
2. **Preprocessing**: Normalize coordinates, scale, and standardize features
3. **Model Training**: Train MLP classifier with controlled experiments (V1-V4)
4. **Evaluation**: Assess model performance on independent test set
5. **Deployment**: Export to ONNX for real-time sign recognition

## Model Versions
- **V1**: Initial deduplicated MLP experiment
- **V2**: Hand-present model experiment
- **V3**: Controlled training experiment (current primary model)
- **V4**: Gradient diagnostic experiment

## Key Files
- `DATASET.md`: Detailed dataset documentation
- `TECHNICAL_DETAILS.md`: Technical implementation details
- `outputs/project_audit/dataset_quality_report.md`: Data quality assessment

## Requirements
- Python 3.10+
- PyTorch, MediaPipe, OpenCV, NumPy, scikit-learn, joblib
- Hand Landmarker model file: `models/hand_landmarker.task`

## Running the Pipeline
1. **Collect data**: `python -m src.data_collection.collect_landmarks`
2. **Train model**: `python -m src.training.train_deduplicated_mlp`
3. **Evaluate**: `python -m src.evaluation.evaluate_model`
4. **Real-time recognition**: `python -m src.recognition.realtime_recognition`
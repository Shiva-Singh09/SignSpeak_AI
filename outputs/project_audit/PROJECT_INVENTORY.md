# SignSpeak AI - Project Inventory

**Inventory date:** 11 November 2026  
**Project root:** `SignSpeak_AI`  
**Purpose:** ISL (Indian Sign Language) alphabet recognition, A-Z.

> This inventory is based on previously shared PowerShell outputs and project records. It is not a fresh scan of the filesystem.

## 1. Main Project Areas

| Folder | Purpose |
|---|---|
| `.venv` | Project-local Python environment |
| `dataset` | Raw, external, and processed datasets |
| `models` | Trained models and hand-landmarker model |
| `src` | Data collection, hand tracking, preprocessing, training, evaluation, and recognition code |
| `outputs` | Evaluation reports, graphs, and other generated results |
| `cache` | Project-local caches |
| `temp` | Temporary files |
| `notebooks` | Notebook workspace |

## 2. Dataset Inventory

Previously reported file counts:

| Dataset location | Files |
|---|---:|
| `dataset/external/RealSign` | 26,004 |
| `dataset/processed/realsign_deduplicated` | 25,988 |
| `dataset/processed/realsign_landmarks` | 4 |
| `dataset/processed/realsign_features` | 3 |
| `dataset/processed/realsign_clean` | 3 |
| `dataset/raw/landmarks.csv` | 1 |
| `dataset/raw/alphabet_landmarks.csv` | 1 |

The self-collected alphabet dataset contains ISL alphabet samples. Earlier collection records reported two samples per alphabet, but initial quality checks found many frames without a detected hand. More data-quality checks and additional signer samples are recommended.

## 3. Model Inventory

Previously reported files in `models`:

| File | Purpose |
|---|---|
| `hand_landmarker.task` | MediaPipe hand detection and landmark model |
| `signspeak_mlp_best.pth` | Earlier best PyTorch checkpoint |
| `signspeak_mlp_v2.pth` | PyTorch model checkpoint |
| `signspeak_mlp_deduplicated_v1.pth` | Deduplicated-data experiment |
| `signspeak_mlp_deduplicated_v2_hand_present.pth` | Hand-present experiment |
| `signspeak_mlp_deduplicated_v3_controlled.pth` | Controlled training experiment |
| `signspeak_mlp_deduplicated_v4_gradient_diagnostic.pth` | Gradient diagnostic experiment |
| V3 ONNX export | Exported representation of the controlled model |

The V3 model uses 132 input features and produces 26 alphabet-class logits.

## 4. Source Code Areas

| Folder | Main responsibility |
|---|---|
| `src/data_collection` | Collect hand landmarks |
| `src/hand_tracking` | Hand tracking and coordinate extraction |
| `src/preprocessing` | Clean data, inspect landmarks, prepare features |
| `src/training` | Train MLP models and export ONNX |
| `src/evaluation` | Dataset audits, model evaluation, comparisons, and reports |
| `src/recognition` | Real-time recognition experiments |
| `src/experiments` | Learning experiments, including backpropagation |

Representative scripts include `collect_landmarks.py`, `hand_tracking_demo.py`, and `train_deduplicated_mlp_v3_controlled.py`.

## 5. Evaluation Outputs

Previously reported evaluation artifacts include:

- Training-history CSV files.
- Validation reports and confusion matrices.
- Duplicate and near-duplicate audit results.
- Hand-detection and handedness audit images.
- Model-comparison summaries.
- Prediction-confidence analysis.

**Large file warning:** `outputs/reports/near_duplicates.csv` was previously reported at approximately 206 MB. Avoid adding this large generated report to a normal Git commit without a deliberate storage plan.

## 6. Environment

Previously reported setup:

- Python 3.12.7 in the project-local `.venv`.
- PyTorch 2.14.1+cu130.
- NumPy 2.5.4.
- MediaPipe 0.10.35.
- OpenCV 5.0.0.
- NVIDIA GeForce RTX 4050 Laptop GPU.
- CUDA availability was successfully tested.

These versions are based on earlier terminal outputs and have not been rechecked while creating this document.

## 7. Documentation

Project documentation includes:

- `README.md`
- `SETUP.md`
- `DATASET.md`
- `TECHNICAL_DETAILS.md`
- `MODEL_EVALUATION.md`
- `DEVELOPMENT_JOURNAL.md`

Additional project audit documents are stored under `outputs/project_audit`.

## 8. Important Notes

- This document records previously reported inventory information, not a fresh filesystem scan.
- File counts and environment details may change as development continues.
- No separate `signspeak_metadata.json` file was found in the earlier JSON inventory.
- Git repository status has not been checked.
- Existing files and model artifacts have not been renamed or deleted as part of this documentation update.

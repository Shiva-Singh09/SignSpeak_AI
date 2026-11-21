# SignSpeak AI - Dataset Documentation

## 1. Dataset Sources

### Primary Dataset: Self-Collected ISL Alphabet Dataset

The project includes a self-collected Indian Sign Language (ISL) alphabet dataset created for static alphabet recognition.

The dataset contains hand landmark data associated with alphabet classes A-Z. Collection quality, signer variation, and correct class labeling are important factors in model performance.

### External Dataset

The project also includes the external RealSign dataset, which is used in separate data preparation and model experiments.

The self-collected dataset and external dataset should be treated as distinct sources. Their samples and evaluation results should not be assumed to be interchangeable.

## 2. Collection Methodology

### Data Collection Process

- **Collection Environment:** Camera-based collection under available lighting conditions.
- **Participants:** Self-collected samples; signer diversity should be increased and documented as additional data becomes available.
- **Signing Style:** Static alphabet signs from A to Z.
- **Repetitions:** Multiple samples per alphabet class.
- **Capture Setup:** A camera source with the hand visible in the captured frame.
- **Landmark Extraction:** MediaPipe Hand Landmarker is used to extract hand landmarks.

### Data Quality Considerations

- Keep the hand visible during capture.
- Avoid excessive occlusion.
- Use sufficient lighting.
- Check that the detected hand matches the intended alphabet label.
- Inspect frames in which no hand is detected.
- Collect additional samples for classes with insufficient variation.

Some early self-collected samples contained a high proportion of frames without detected hands. These samples require quality checks before being used for reliable training.

## 3. Dataset Folder Structure

The project dataset is organized into the following main directories:

```text
dataset/
|
+-- raw/
|   +-- landmarks.csv
|   +-- alphabet_landmarks.csv
|
+-- processed/
|   +-- realsign_deduplicated/
|   +-- realsign_landmarks/
|   +-- realsign_features/
|   +-- realsign_clean/
|
+-- external/
|   +-- RealSign/
|
+-- labels/
```

The exact contents may change as additional collection and preprocessing experiments are performed.

Generated intermediate files should be kept inside the relevant dataset directories.

## 4. Sample Counts and Dataset Splits

Exact sample counts depend on the dataset version and preparation script.

Do not use placeholder values as actual dataset statistics. Verify counts from the corresponding dataset files, split-generation scripts, and evaluation reports before publishing results.

The project uses separate training, validation, and test data for model development and evaluation.

| Split | Sample count |
|---|---|
| Training | Verify from the generated split |
| Validation | Verify from the generated split |
| Test | Verify from the generated split |

Validation results and test results must be reported separately.

## 5. Class Distribution

The alphabet classification task contains 26 classes:

A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z.

Class distribution should be checked using the actual dataset rather than assuming that every class has equal representation.

Any imbalance, missing samples, or low-quality samples should be documented using dataset analysis reports.

## 6. Labeling

- **Label Scheme:** Alphabetical A-Z mapping.
- **Label Index:** The intended class mapping is 0-25, corresponding to A-Z.
- **Labeling Criteria:** Samples are assigned according to their intended ISL alphabet class.
- **Quality Checks:** Samples should be inspected for incorrect labels, missing-hand frames, and inconsistent landmark data.

The class-to-index mapping used during training must match the mapping used during evaluation and inference.

## 7. Landmark Structure

### MediaPipe Hand Landmarks

MediaPipe Hand Landmarker detects 21 landmarks per hand.

Each landmark contains three coordinates:

- **x:** Normalized horizontal coordinate relative to image width.
- **y:** Normalized vertical coordinate relative to image height.
- **z:** Relative depth coordinate.

Each hand therefore contributes 63 coordinate values:

21 landmarks x 3 coordinates = 63 features.

### Landmark Indices and Names

| Index | Landmark Name | Description |
|---:|---|---|
| 0 | Wrist | Wrist point |
| 1 | Thumb CMC | Thumb base |
| 2 | Thumb MCP | Thumb metacarpophalangeal joint |
| 3 | Thumb IP | Thumb interphalangeal joint |
| 4 | Thumb Tip | Thumb endpoint |
| 5 | Index MCP | Index finger base |
| 6 | Index PIP | Index proximal interphalangeal joint |
| 7 | Index DIP | Index distal interphalangeal joint |
| 8 | Index Tip | Index finger tip |
| 9 | Middle MCP | Middle finger base |
| 10 | Middle PIP | Middle finger proximal interphalangeal joint |
| 11 | Middle DIP | Middle finger distal interphalangeal joint |
| 12 | Middle Tip | Middle finger tip |
| 13 | Ring MCP | Ring finger base |
| 14 | Ring PIP | Ring finger proximal interphalangeal joint |
| 15 | Ring DIP | Ring finger distal interphalangeal joint |
| 16 | Ring Tip | Ring finger tip |
| 17 | Pinky MCP | Pinky finger base |
| 18 | Pinky PIP | Pinky proximal interphalangeal joint |
| 19 | Pinky DIP | Pinky distal interphalangeal joint |
| 20 | Pinky Tip | Pinky finger tip |

## 8. Feature Definitions

The final model input contains exactly 132 float32 features.

### 8.1 Per-Hand Features

Each hand contributes 63 features:

- Wrist XYZ coordinates: 3 values.
- Landmarks 1-20 relative to the wrist: 60 values.

Total:

21 landmarks x 3 coordinates = 63 features per hand.

### 8.2 Two-Hand Feature Layout

The first 126 features contain coordinates for two hand slots:

- Left hand: features 0-62.
- Right hand: features 63-125.

If a hand is absent, its corresponding 63 coordinate features are filled with zeros.

The left-hand and right-hand ordering must remain consistent across data preparation, training, evaluation, and inference.

### 8.3 Hand Presence and Handedness Flags

The final six features are flags:

| Feature Index | Flag | Meaning |
|---:|---|---|
| 126 | hand_0_present | 1 if hand slot 0 is present |
| 127 | hand_1_present | 1 if hand slot 1 is present |
| 128 | hand_0_is_left | 1 if hand slot 0 is left |
| 129 | hand_0_is_right | 1 if hand slot 0 is right |
| 130 | hand_1_is_left | 1 if hand slot 1 is left |
| 131 | hand_1_is_right | 1 if hand slot 1 is right |

These six flags are not standardized.

## 9. Preprocessing Operations

The coordinate preprocessing pipeline includes the following operations:

1. **Landmark Normalization:** The wrist is used as the origin for wrist-relative landmark coordinates.
2. **Scale Normalization:** Calculate the maximum Euclidean 3D distance of the 20 wrist-relative landmarks from the wrist. Divide the relative coordinates by this scale. If the scale is less than or equal to 1e-8, use zeros.
3. **Hand Assignment:** Store left-hand features first (0-62) and right-hand features second (63-125).
4. **Flag Assignment:** Store the six hand-presence and handedness flags in positions 126-131.
5. **Standardization:** Apply `(feature - mean) / std` to the first 126 coordinate features only. If std is less than 1e-8, use 1.

The scaler statistics must be derived from the appropriate training data and reused consistently during evaluation and inference.

Do not standardize the six hand-presence and handedness flags.

## 10. Data Quality and Leakage Prevention

Dataset quality is essential for reliable evaluation.

Important checks include:

- Verify that every sample has the expected feature count.
- Inspect missing-hand frames.
- Check class labels and class distribution.
- Detect exact duplicate samples.
- Analyse near-duplicate samples.
- Keep related samples from leaking across training and evaluation splits.
- Verify that preprocessing statistics are calculated from training data only.
- Evaluate performance on independent data wherever possible.

High validation accuracy alone does not establish generalization to unseen signers or different capture conditions.

## 11. Important Notes

- The current classification task covers static ISL alphabet classes A-Z.
- It does not represent complete ISL language recognition.
- Landmark quality depends on hand visibility and detection conditions.
- The 132-feature layout is part of the model input contract.
- Any change to feature ordering or preprocessing must be checked against the trained model.
- Dataset counts and evaluation metrics should always be taken from the relevant actual dataset version and report.

---

**SignSpeak AI - Dataset Documentation**

# SignSpeak AI - Project Review

**Review date:** 11 November 2026

## 1. Overall Project Status

SignSpeak AI is an Indian Sign Language (ISL) alphabet-recognition project.

The Python-based AI pipeline has progressed through data collection, hand-landmark extraction, preprocessing, model training, evaluation, error analysis and ONNX export.

The current development focus is improving dataset quality, validating the model-input contract and strengthening recognition reliability.

## 2. What Is Working Well

- A project-local Python virtual environment has been configured.
- GPU-based PyTorch execution was tested successfully.
- MediaPipe hand tracking and webcam capture were tested.
- Data collection and preprocessing scripts have been developed.
- Multiple MLP model experiments have been trained.
- Dataset duplicate and near-duplicate analysis has been performed.
- Evaluation reports and confusion matrices are available.
- A V3 model has been exported to ONNX.
- An earlier ONNX comparison reported matching predictions for 512 synthetic inputs.

These points are based on previously shared outputs and project records, not a fresh execution during this review.

## 3. Model Evaluation Findings

Two different types of results must be kept separate.

### Controlled Validation Comparison

On the same 2,284 hand-present validation samples:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| V1 | 99.606% | 99.613% |
| V2 | 99.869% | 99.873% |
| V3 Controlled | 99.825% | 99.832% |

V1 and V3 provide the more controlled comparison because they use the same architecture and feature ordering, with different training-data conditions.

V2 also changes the order of BatchNorm and ReLU, so its result is not a pure data-filtering comparison.

### Separate Test Result

The previously reported test metrics for `signspeak_mlp_best.pth` are:

- Test samples: 4,630
- Accuracy: 92.894%
- Macro precision: 93.842%
- Macro recall: 92.490%
- Macro F1: 92.494%

These results belong to a different checkpoint and evaluation run. Do not present them as the V3 validation result.

## 4. Main Risks and Limitations

### Dataset Quality

- Early self-collected alphabet samples contained many frames without a detected hand.
- More samples and signer diversity are needed.
- Dataset splits must be checked for duplicate or near-duplicate leakage.
- RealSign data and its class mapping should be verified before combining it with other datasets.

### Model Generalization

- High validation accuracy does not guarantee similar accuracy in real-world conditions.
- Performance should be tested with unseen signers, different lighting, backgrounds, camera positions and hand orientations.
- Confusing alphabet classes need targeted testing.

### Recognition Pipeline

- A successful hand-tracking demonstration does not establish that the complete recognition pipeline is reliable.
- The exact feature ordering and preprocessing must remain consistent between training and inference.
- ONNX numerical parity on synthetic inputs does not prove correct predictions on real-world inputs.
- Real-time prediction speed and stability require further evaluation.

### Project Maintenance

- The near-duplicate report is approximately 206 MB and needs a deliberate storage strategy.
- Model files, datasets, temporary files and generated reports should not be committed blindly.
- Git repository status has not been checked.

## 5. Recommended Development Priorities

1. **Freeze the model contract:** Document the exact 132-feature input order, hand-slot assignment, coordinate normalization, standardization and output class order.
2. **Strengthen dataset validation:** Verify labels, hand presence, duplicate leakage and signer-independent evaluation.
3. **Improve generalization:** Collect more varied samples and test with unseen signers.
4. **Continue error analysis:** Investigate confusing alphabet classes, incorrect predictions and prediction confidence.
5. **Validate the complete recognition pipeline:** Test camera input, landmark extraction, feature preparation and model inference together.
6. **Evaluate performance:** Measure prediction speed, stability and behavior under different lighting and hand positions.
7. **Maintain documentation:** Update technical details, evaluation results and development history as the project changes.

## 6. Recommended AI Recognition Pipeline

Camera input  
→ Image rotation and RGB conversion  
→ Hand landmark detection  
→ 132-feature extraction  
→ Model-compatible input preparation  
→ ONNX inference  
→ Predicted ISL alphabet

The inference pipeline must reproduce the Python feature ordering and preprocessing exactly.

The exported V3 ONNX model embeds its coordinate standardization. An inference pipeline using this model should provide the expected raw 132-feature vector rather than applying the same standardization a second time.

## 7. Final Assessment

**AI foundation:** Substantial progress; training and evaluation artifacts are available.

**Model reliability:** Promising validation results, but more independent real-world testing is required.

**Dataset quality:** Further signer diversity, data-quality checks and leakage prevention are important.

**Recognition pipeline:** Core components exist, but complete real-time behavior requires further verification.

**Project readiness:** The AI foundation is established. The next priority is structured dataset validation, model-contract verification and end-to-end recognition testing.

## 8. Review Limitations

This review uses previously supplied terminal outputs, evaluation summaries and project history.

It is not a fresh execution of the training or evaluation workflows.

Any unverified historical result should be checked before being used in a formal presentation.

---

Document status: Project review based on available project records. Metrics and historical observations are reported as previously supplied, not as newly executed tests.

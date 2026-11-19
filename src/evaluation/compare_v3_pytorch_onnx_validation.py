import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models" / "signspeak_mlp_deduplicated_v3_controlled.pth"
ONNX_MODEL = ROOT / "models" / "signspeak_mlp_deduplicated_v3_android.onnx"
VALIDATION_CSV = (
    ROOT / "dataset" / "processed" / "realsign_deduplicated"
    / "features_hand_present" / "validation_features.csv"
)


class SignSpeakMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(132, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),
            nn.Linear(128, 26),
        )

    def forward(self, x):
        return self.network(x)


def main():
    print("=" * 68)
    print("SIGNSPEAK - REAL VALIDATION PYTORCH VS ONNX PARITY")
    print("=" * 68)

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )

    with VALIDATION_CSV.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)

    if len(header) != 134:
        raise ValueError(f"Expected 134 CSV columns, found {len(header)}")

    if len(rows) == 0:
        raise ValueError("Validation CSV contains no samples.")

    if any(len(row) != 134 for row in rows):
        raise ValueError("Inconsistent CSV row length detected.")

    labels = [row[0] for row in rows]
    features = np.asarray(
        [[float(value) for value in row[2:]] for row in rows],
        dtype=np.float32,
    )

    if features.shape[1] != 132:
        raise ValueError(f"Expected 132 features, got {features.shape[1]}")

    if not np.isfinite(features).all():
        raise ValueError("Validation features contain non-finite values.")

    model = SignSpeakMLP()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    mean = torch.as_tensor(
        checkpoint["mean"], dtype=torch.float32
    ).reshape(1, 126)
    std = torch.as_tensor(
        checkpoint["std"], dtype=torch.float32
    ).reshape(1, 126)

    x = torch.from_numpy(features)
    normalized = x.clone()
    normalized[:, :126] = (
        normalized[:, :126] - mean
    ) / std

    with torch.inference_mode():
        pytorch_logits = model(normalized).numpy()

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )
    onnx_logits = session.run(
        ["logits"], {"features": features}
    )[0]

    if pytorch_logits.shape != onnx_logits.shape:
        raise ValueError("PyTorch and ONNX output shapes do not match.")

    difference = np.abs(pytorch_logits - onnx_logits)
    max_index = np.unravel_index(np.argmax(difference), difference.shape)

    pytorch_predictions = pytorch_logits.argmax(axis=1)
    onnx_predictions = onnx_logits.argmax(axis=1)

    classes = list(checkpoint["classes"])
    agreement = np.mean(
        pytorch_predictions == onnx_predictions
    ) * 100

    print("Validation samples:", len(features))
    print("Input shape:", features.shape)
    print("PyTorch output shape:", pytorch_logits.shape)
    print("ONNX output shape:", onnx_logits.shape)
    print("Maximum absolute difference:", float(difference.max()))
    print("Mean absolute difference:", float(difference.mean()))
    print("Prediction agreement:", f"{agreement:.4f}%")
    print(
        "Largest difference location:",
        f"sample={max_index[0]}, class={classes[max_index[1]]}",
    )
    print("Actual label at that sample:", labels[max_index[0]])
    print(
        "PyTorch prediction:",
        classes[pytorch_predictions[max_index[0]]],
    )
    print(
        "ONNX prediction:",
        classes[onnx_predictions[max_index[0]]],
    )

    passed = (
        np.allclose(
            pytorch_logits,
            onnx_logits,
            rtol=1e-4,
            atol=1e-5,
        )
        and agreement == 100.0
    )

    print("-" * 68)
    print("NUMERICAL PARITY:", "PASSED" if passed else "FAILED")
    print("=" * 68)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

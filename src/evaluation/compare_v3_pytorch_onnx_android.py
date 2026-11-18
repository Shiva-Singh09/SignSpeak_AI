from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models" / "signspeak_mlp_deduplicated_v3_controlled.pth"
ONNX_MODEL = ROOT / "models" / "signspeak_mlp_deduplicated_v3_android.onnx"


class V3Model(nn.Module):
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
    print("=" * 65)
    print("SIGNSPEAK - PYTORCH VS ANDROID ONNX PARITY TEST")
    print("=" * 65)

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )

    model = V3Model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = torch.as_tensor(checkpoint["mean"], dtype=torch.float32)
    std = torch.as_tensor(checkpoint["std"], dtype=torch.float32)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    rng = np.random.default_rng(42)
    features = rng.uniform(-1.0, 1.0, size=(512, 132)).astype(np.float32)

    # Generate valid presence and handedness flag combinations.
    for i in range(len(features)):
        mode = i % 4
        if mode == 0:
            flags = [0, 0, 0, 0, 0, 0]
        elif mode == 1:
            flags = [1, 0, 1, 0, 0, 0]
        elif mode == 2:
            flags = [1, 0, 0, 1, 0, 0]
        else:
            flags = [1, 1, 1, 0, 0, 1]
        features[i, 126:] = flags

    x = torch.from_numpy(features)
    normalized = x.clone()
    normalized[:, :126] = (normalized[:, :126] - mean) / std

    with torch.inference_mode():
        pytorch_logits = model(normalized).numpy()

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )
    onnx_logits = session.run(
        ["logits"], {"features": features}
    )[0]

    difference = np.abs(pytorch_logits - onnx_logits)
    max_diff = float(difference.max())
    mean_diff = float(difference.mean())

    pytorch_predictions = pytorch_logits.argmax(axis=1)
    onnx_predictions = onnx_logits.argmax(axis=1)
    agreement = float(
        np.mean(pytorch_predictions == onnx_predictions) * 100
    )

    print("Test samples:", len(features))
    print("PyTorch output shape:", pytorch_logits.shape)
    print("ONNX output shape:", onnx_logits.shape)
    print("Maximum absolute difference:", max_diff)
    print("Mean absolute difference:", mean_diff)
    print("Prediction agreement:", f"{agreement:.2f}%")

    passed = (
        np.allclose(
            pytorch_logits, onnx_logits, rtol=1e-4, atol=1e-5
        )
        and agreement == 100.0
    )

    print("-" * 65)
    print("PARITY TEST:", "PASSED" if passed else "FAILED")
    print("=" * 65)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

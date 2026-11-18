from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "models" / "signspeak_mlp_deduplicated_v3_controlled.pth"
OUTPUT_PATH = ROOT / "models" / "signspeak_mlp_deduplicated_v3_android.onnx"


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


class AndroidInferenceModel(nn.Module):
    """Includes the exact training scaler before the trained MLP."""

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean.reshape(1, 126))
        self.register_buffer("std", std.reshape(1, 126))

    def forward(self, x):
        coordinates = (x[:, :126] - self.mean) / self.std
        flags = x[:, 126:]
        normalized = torch.cat((coordinates, flags), dim=1)
        return self.model(normalized)


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT_PATH}"
        )

    print("=" * 65)
    print("SIGNSPEAK - V3 ANDROID ONNX EXPORT")
    print("=" * 65)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("architecture") != "132-256-128-26":
        raise ValueError("Unexpected checkpoint architecture.")

    if checkpoint.get("input_size") != 132:
        raise ValueError("Expected 132 input features.")

    if checkpoint.get("num_classes") != 26:
        raise ValueError("Expected 26 output classes.")

    classes = list(checkpoint["classes"])
    if len(classes) != 26:
        raise ValueError("Checkpoint must contain 26 class labels.")

    mean = torch.as_tensor(checkpoint["mean"], dtype=torch.float32).flatten()
    std = torch.as_tensor(checkpoint["std"], dtype=torch.float32).flatten()

    if mean.shape != (126,) or std.shape != (126,):
        raise ValueError("Scaler must contain exactly 126 coordinate values.")

    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Scaler contains non-finite values.")

    if (std <= 0).any():
        raise ValueError("Scaler standard deviations must be positive.")

    base_model = SignSpeakMLP()
    base_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    base_model.eval()

    inference_model = AndroidInferenceModel(base_model, mean, std)
    inference_model.eval()

    # A deterministic sample is sufficient to trace the inference graph.
    dummy_input = torch.zeros((2, 132), dtype=torch.float32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            inference_model,
            dummy_input,
            str(OUTPUT_PATH),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    exported = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(exported)

    metadata = {
        "project": "SignSpeak AI",
        "architecture": "132-256-128-26",
        "classes": ",".join(str(label) for label in classes),
        "input_features": "132",
        "coordinate_features": "126",
        "preprocessing": "embedded coordinate standardization; six flags unchanged",
        "output": "raw_logits; apply softmax for probabilities",
    }

    for key, value in metadata.items():
        entry = exported.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.save(exported, str(OUTPUT_PATH))

    # Check the saved model again after metadata is added.
    onnx.checker.check_model(onnx.load(str(OUTPUT_PATH)))

    print("Export successful.")
    print("Architecture:", checkpoint["architecture"])
    print("Input shape: [batch_size, 132]")
    print("Output shape: [batch_size, 26]")
    print("Classes:", " ".join(str(label) for label in classes))
    print("Opset: 17")
    print("Scaler: embedded in ONNX graph")
    print("ONNX validation: PASSED")
    print("Output:", OUTPUT_PATH)
    print("Size (MB):", round(OUTPUT_PATH.stat().st_size / (1024 * 1024), 3))


if __name__ == "__main__":
    main()

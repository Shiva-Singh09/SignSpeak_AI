
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS = [
    PROJECT_ROOT / "models" / "signspeak_mlp_deduplicated_v1.pth",
    PROJECT_ROOT / "models" / "signspeak_mlp_deduplicated_v2_hand_present.pth",
    PROJECT_ROOT / "models" / "signspeak_mlp_deduplicated_v3_controlled.pth",
]

for checkpoint_path in CHECKPOINTS:

    print("\n" + "=" * 65)
    print(checkpoint_path.name)
    print("=" * 65)

    if not checkpoint_path.exists():
        print("CHECKPOINT NOT FOUND")
        continue

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    print("Type:", type(checkpoint).__name__)

    if not isinstance(checkpoint, dict):
        print("Not a dictionary checkpoint.")
        continue

    print("\nStored keys:")

    for key, value in checkpoint.items():

        if key == "model_state_dict":
            print("\nMODEL LAYERS:")

            for layer_name, tensor in value.items():
                print(f"{layer_name}: {tuple(tensor.shape)}")

        elif key == "class_to_idx":
            print("\nCLASS MAPPING:")
            print(value)

        elif key == "classes":
            print("\nCLASSES:")
            print(value)

        elif key == "feature_columns":
            print("\nFEATURE COLUMNS:")
            print("Count:", len(value))
            print("First 5:", value[:5])
            print("Last 8:", value[-8:])

        elif hasattr(value, "shape"):
            print(f"{key}: Array {value.shape}")

        else:
            print(f"{key}: {str(value)[:200]}")

print("\nCHECKPOINT INSPECTION COMPLETED")


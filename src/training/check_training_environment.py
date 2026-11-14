import torch

print("PYTORCH TRAINING ENVIRONMENT")
print("-" * 40)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(
        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
    ), "GB")

    test = torch.randn(1000, 132, device="cuda")
    result = test @ test.T

    print("GPU computation:", result.device)
    print("GPU test: SUCCESS")
else:
    print("GPU test: CPU fallback only")

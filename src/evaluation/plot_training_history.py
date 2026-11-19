import csv
from pathlib import Path

import matplotlib.pyplot as plt

path = Path("outputs/reports/mlp_training_history.csv")

epochs = []
train_loss = []
val_loss = []
train_accuracy = []
val_accuracy = []

with path.open("r", encoding="utf-8-sig", newline="") as file:
    for row in csv.DictReader(file):
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["validation_loss"]))
        train_accuracy.append(float(row["train_accuracy"]) * 100)
        val_accuracy.append(float(row["validation_accuracy"]) * 100)

output = Path("outputs/graphs")
output.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SignSpeak AI - Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output / "mlp_loss_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(epochs, train_accuracy, label="Training Accuracy")
plt.plot(epochs, val_accuracy, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("SignSpeak AI - Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output / "mlp_accuracy_curve.png", dpi=150)
plt.close()

print("Graphs saved:")
print(output / "mlp_loss_curve.png")
print(output / "mlp_accuracy_curve.png")

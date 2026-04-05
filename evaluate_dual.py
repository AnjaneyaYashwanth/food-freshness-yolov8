import torch
from torch.utils.data import DataLoader
from dataset_loader import DualStreamDataset
from models.dual_stream_model import DualStreamModel

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------
# Settings
# ----------------------------
MODEL_PATH = "best_dual_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["fresh", "ripe", "overripe"]

# ----------------------------
# Dataset
# ----------------------------
test_dataset = DualStreamDataset("data/processed", "test")
test_loader = DataLoader(test_dataset, batch_size=4)

print("Test samples:", len(test_dataset))

# ----------------------------
# Load Model
# ----------------------------
model = DualStreamModel(num_classes=3, dropout=0.285)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# ----------------------------
# Evaluation
# ----------------------------
correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for rgb, edge, lbp, labels in test_loader:
        rgb = rgb.to(device)
        edge = edge.to(device)
        lbp = lbp.to(device)
        labels = labels.to(device)

        outputs = model(rgb, edge, lbp)
        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ----------------------------
# Accuracy
# ----------------------------
accuracy = 100 * correct / total
print(f"\n✅ Test Accuracy: {accuracy:.2f}%")

# ----------------------------
# Classification Report
# ----------------------------
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)

# Create images folder if not exists
os.makedirs("images", exist_ok=True)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix - Dual Stream")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(np.arange(len(classes)), classes)
plt.yticks(np.arange(len(classes)), classes)

# Annotate values
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()

# Save image
plt.savefig("images/confusion_matrix_dual.png")

print("\n📊 Confusion matrix saved → images/confusion_matrix_dual.png")
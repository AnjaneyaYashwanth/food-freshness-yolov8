import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from dataset_loader import DualStreamDataset
from models.dual_stream_model import DualStreamModel

# ----------------------------
# Settings
# ----------------------------
MODEL_PATH = "best_dual_model.pth"
BATCH_SIZE = 4
classes = ["fresh", "ripe", "overripe"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Dataset (TEST SET)
# ----------------------------
test_dataset = DualStreamDataset("data/processed", "test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("Test samples:", len(test_dataset))

# ----------------------------
# Load Model
# ----------------------------
model = DualStreamModel(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# ----------------------------
# Collect Predictions
# ----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for rgb, edge, lbp, labels in test_loader:
        rgb = rgb.to(device)
        edge = edge.to(device)
        lbp = lbp.to(device)

        outputs = model(rgb, edge, lbp)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:\n", cm)

# ----------------------------
# Plot
# ----------------------------
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="Blues")
plt.title("Dual-Stream Confusion Matrix")
plt.savefig("dual_confusion_matrix.png")
plt.show()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset_loader import DualStreamDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Dataset
# ----------------------------
val_dataset = DualStreamDataset(base_dir="data/processed", split="val")
val_loader = DataLoader(val_dataset, batch_size=8)

class_names = ["fresh", "ripe", "overripe"]

# ----------------------------
# Load Model
# ----------------------------
yolo = YOLO("yolov8n-cls.pt")
model = yolo.model.to(device)

# Modify first layer (5-channel)
old_conv = model.model[0].conv

new_conv = nn.Conv2d(
    in_channels=5,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=(old_conv.bias is not None)
)

with torch.no_grad():
    new_conv.weight[:, :3] = old_conv.weight
    nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode='fan_out', nonlinearity='relu')

model.model[0].conv = new_conv

# Modify classifier head
model.model[-1].linear = nn.Linear(
    model.model[-1].linear.in_features,
    3
)

# Load trained weights
model.load_state_dict(torch.load("yolo_5ch_food.pth", weights_only=True))
model = model.to(device)
model.eval()

# ----------------------------
# Collect Predictions
# ----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)

print("Confusion Matrix:\n", cm)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
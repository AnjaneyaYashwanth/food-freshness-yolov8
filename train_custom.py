import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import YOLO
import matplotlib.pyplot as plt

from dataset_loader import DualStreamDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# BEST PARAMS FROM GWO
# ----------------------------
BEST_LR = 0.0005249542093026305
BEST_BATCH = 4

# ----------------------------
# Dataset
# ----------------------------
train_dataset = DualStreamDataset("data/processed", "train")
val_dataset = DualStreamDataset("data/processed", "val")

train_loader = DataLoader(train_dataset, batch_size=BEST_BATCH, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BEST_BATCH)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))

# ----------------------------
# Model
# ----------------------------
yolo = YOLO("yolov8n-cls.pt")
model = yolo.model.to(device)

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

model.model[-1].linear = nn.Linear(
    model.model[-1].linear.in_features,
    3
)

model = model.to(device)

# ----------------------------
# Training Setup
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=BEST_LR)

train_losses = []
val_accuracies = []

# 🔥 Track best model
best_acc = 0

# ----------------------------
# Training
# ----------------------------
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_losses.append(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # ----------------------------
    # Validation
    # ----------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, pred = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = 100 * correct / total
    val_accuracies.append(acc)

    print(f"Validation Accuracy: {acc:.2f}%")

    # 🔥 SAVE BEST MODEL
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("🔥 Best model updated!")

# ----------------------------
# FINAL SAVE (optional)
# ----------------------------
torch.save(model.state_dict(), "last_model.pth")

print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")

# ----------------------------
# Graphs
# ----------------------------
plt.figure()
plt.plot(train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("loss_curve_optimized.png")

plt.figure()
plt.plot(val_accuracies, marker='o')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.savefig("accuracy_curve_optimized.png")

print("📊 Graphs saved!")
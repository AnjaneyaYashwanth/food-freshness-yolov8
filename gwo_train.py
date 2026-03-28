import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import YOLO
import random

from dataset_loader import DualStreamDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset
# ----------------------------
train_dataset = DualStreamDataset("data/processed", "train")
val_dataset = DualStreamDataset("data/processed", "val")

# ----------------------------
# Model Builder
# ----------------------------
def build_model():
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

    return model.to(device)

# ----------------------------
# Fitness Function
# ----------------------------
def evaluate_model(lr, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # short training (fast evaluation)
    for _ in range(3):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # validation
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

    return correct / total

# ----------------------------
# GWO
# ----------------------------
num_wolves = 5
iterations = 5

lr_range = [1e-5, 1e-3]
batch_options = [4, 8, 16]

wolves = []

for _ in range(num_wolves):
    wolves.append({
        "lr": random.uniform(*lr_range),
        "batch": random.choice(batch_options),
        "score": 0
    })

for iter in range(iterations):
    print(f"\nIteration {iter+1}")

    for wolf in wolves:
        wolf["score"] = evaluate_model(wolf["lr"], wolf["batch"])
        print(f"LR: {wolf['lr']:.6f}, Batch: {wolf['batch']}, Acc: {wolf['score']:.4f}")

    wolves = sorted(wolves, key=lambda x: x["score"], reverse=True)

    alpha, beta, delta = wolves[:3]

    for wolf in wolves[3:]:
        wolf["lr"] = (alpha["lr"] + beta["lr"] + delta["lr"]) / 3
        wolf["batch"] = random.choice(batch_options)

best = wolves[0]

print("\n🔥 BEST HYPERPARAMETERS FOUND:")
print(f"Learning Rate: {best['lr']}")
print(f"Batch Size: {best['batch']}")
print(f"Validation Accuracy: {best['score']*100:.2f}%")
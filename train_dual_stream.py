import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset_loader import DualStreamDataset
from models.dual_stream_model import DualStreamModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Hyperparameters (same as GWO best)
# ----------------------------
BEST_LR = 0.0004
BEST_BATCH = 4
EPOCHS = 20

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
model = DualStreamModel(num_classes=3).to(device)

# ----------------------------
# Training Setup
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=BEST_LR)

train_losses = []
val_accuracies = []
best_acc = 0

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for rgb, edge, lbp, labels in train_loader:
        rgb = rgb.to(device)
        edge = edge.to(device)
        lbp = lbp.to(device)
        labels = labels.to(device)

        outputs = model(rgb, edge, lbp)
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
        for rgb, edge, lbp, labels in val_loader:
            rgb = rgb.to(device)
            edge = edge.to(device)
            lbp = lbp.to(device)
            labels = labels.to(device)

            outputs = model(rgb, edge, lbp)
            _, pred = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = 100 * correct / total
    val_accuracies.append(acc)

    print(f"Validation Accuracy: {acc:.2f}%")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_dual_model.pth")
        print("🔥 Best dual-stream model updated!")

# ----------------------------
# Save final model
# ----------------------------
torch.save(model.state_dict(), "last_dual_model.pth")

print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")

# ----------------------------
# Graphs
# ----------------------------
plt.figure()
plt.plot(train_losses, marker='o')
plt.title("Dual Stream Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("dual_loss_curve.png")

plt.figure()
plt.plot(val_accuracies, marker='o')
plt.title("Dual Stream Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.savefig("dual_accuracy_curve.png")

print("📊 Graphs saved!")
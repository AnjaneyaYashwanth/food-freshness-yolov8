# %%
import torch
import torch.nn as nn
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
# Load pretrained YOLOv8 classification model
yolo = YOLO("yolov8n-cls.pt")
model = yolo.model  # Extract PyTorch model
model = model.to(device)

print("Original first layer:", model.model[0])

# %%
# Get original conv layer
old_conv = model.model[0].conv

# Create new conv layer (5 input channels)
new_conv = nn.Conv2d(
    in_channels=5,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=(old_conv.bias is not None)
)

# %%
# Copy pretrained weights
with torch.no_grad():
    # Copy RGB weights
    new_conv.weight[:, :3, :, :] = old_conv.weight

    # Initialize extra channels (Edge + LBP)
    nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

    # Copy bias if exists
    if old_conv.bias is not None:
        new_conv.bias = old_conv.bias

# %%
# Replace layer
model.model[0].conv = new_conv

print("Modified first layer:", model.model[0])

# %%
# Send to device again (safe)
model = model.to(device)

# %%
# Quick sanity check
dummy_input = torch.randn(1, 5, 640, 640).to(device)
output = model(dummy_input)

if isinstance(output, tuple):
    output = output[0]

print("Forward pass successful! Output shape:", output.shape)
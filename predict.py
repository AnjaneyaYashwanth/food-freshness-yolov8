import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# CLASS NAMES
# ----------------------------
class_names = ["fresh", "ripe", "overripe"]

# ----------------------------
# LOAD MODEL
# ----------------------------
yolo = YOLO("yolov8n-cls.pt")
model = yolo.model.to(device)

# Modify first layer (5 channels)
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
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model = model.to(device)
model.eval()

# ----------------------------
# FEATURE GENERATION
# ----------------------------

def compute_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 100, 200)
    return edge

def compute_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = np.zeros_like(gray)

    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i, j]
            binary = [
                gray[i-1, j-1] > center,
                gray[i-1, j] > center,
                gray[i-1, j+1] > center,
                gray[i, j+1] > center,
                gray[i+1, j+1] > center,
                gray[i+1, j] > center,
                gray[i+1, j-1] > center,
                gray[i, j-1] > center
            ]
            value = sum([b << idx for idx, b in enumerate(binary)])
            lbp[i, j] = value

    return lbp

# ----------------------------
# PREPROCESS IMAGE
# ----------------------------
def preprocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))

    rgb = image / 255.0
    edge = compute_edge(image) / 255.0
    lbp = compute_lbp(image) / 255.0

    rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)
    edge = torch.tensor(edge, dtype=torch.float32).unsqueeze(0)
    lbp = torch.tensor(lbp, dtype=torch.float32).unsqueeze(0)

    combined = torch.cat([rgb, edge, lbp], dim=0)  # [5, H, W]

    return combined.unsqueeze(0)  # add batch dim

# ----------------------------
# PREDICT FUNCTION
# ----------------------------
def predict(image_path):
    input_tensor = preprocess(image_path).to(device)

    with torch.no_grad():
        output = model(input_tensor)

        if isinstance(output, tuple):
            output = output[0]

        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    print("\nClass Probabilities:")
    for i, cls in enumerate(class_names):
        print(f"{cls}: {probs[0][i].item():.4f}")
    print("\nFinal Prediction:", class_names[pred_class])

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")

    args = parser.parse_args()

    predict(args.image)
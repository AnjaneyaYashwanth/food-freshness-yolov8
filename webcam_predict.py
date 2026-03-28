import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Class names
# ----------------------------
class_names = ["fresh", "ripe", "overripe"]

# ----------------------------
# Load model
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

# Modify classifier
model.model[-1].linear = nn.Linear(
    model.model[-1].linear.in_features,
    3
)

# Load trained weights
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model = model.to(device)
model.eval()

# ----------------------------
# Feature functions
# ----------------------------
def compute_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

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
# Preprocess
# ----------------------------
def preprocess(frame):
    image = cv2.resize(frame, (640, 640))

    rgb = image / 255.0
    edge = compute_edge(image) / 255.0
    lbp = compute_lbp(image) / 255.0

    rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)
    edge = torch.tensor(edge, dtype=torch.float32).unsqueeze(0)
    lbp = torch.tensor(lbp, dtype=torch.float32).unsqueeze(0)

    combined = torch.cat([rgb, edge, lbp], dim=0)

    return combined.unsqueeze(0)

# ----------------------------
# Webcam loop
# ----------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame).to(device)

    with torch.no_grad():
        output = model(input_tensor)

        if isinstance(output, tuple):
            output = output[0]

        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    label = f"{class_names[pred_class]} ({confidence:.2f})"

    # Display on frame
    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Food Freshness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import torch
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from models.dual_stream_model import DualStreamModel

# ----------------------------
# Settings
# ----------------------------
MODEL_PATH = "best_dual_model.pth"
IMAGE_PATH = "data/test/fresh/IMG_20241030_195843539.jpg"

classes = ["fresh", "ripe", "overripe"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Model
# ----------------------------
model = DualStreamModel(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# ----------------------------
# Load Image
# ----------------------------
rgb = cv2.imread(IMAGE_PATH)

if rgb is None:
    raise ValueError(" Image not found. Check path!")

rgb = cv2.resize(rgb, (640, 640))

# ----------------------------
# Create EDGE
# ----------------------------
edge = cv2.Canny(rgb, 100, 200)

# ----------------------------
# Create LBP
# ----------------------------
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
lbp = np.zeros_like(gray)

for i in range(1, gray.shape[0]-1):
    for j in range(1, gray.shape[1]-1):
        center = gray[i, j]
        code = 0
        code |= (gray[i-1, j-1] > center) << 7
        code |= (gray[i-1, j] > center) << 6
        code |= (gray[i-1, j+1] > center) << 5
        code |= (gray[i, j+1] > center) << 4
        code |= (gray[i+1, j+1] > center) << 3
        code |= (gray[i+1, j] > center) << 2
        code |= (gray[i+1, j-1] > center) << 1
        code |= (gray[i, j-1] > center) << 0
        lbp[i, j] = code

# ----------------------------
# Normalize
# ----------------------------
rgb = rgb / 255.0
edge = edge / 255.0
lbp = lbp / 255.0

# ----------------------------
# Convert to Tensor
# ----------------------------
rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
edge = torch.tensor(edge, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
lbp = torch.tensor(lbp, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

rgb, edge, lbp = rgb.to(device), edge.to(device), lbp.to(device)

# ----------------------------
# Prediction
# ----------------------------
with torch.no_grad():
    outputs = model(rgb, edge, lbp)

    probs = torch.softmax(outputs, dim=1)
    confidence, pred = torch.max(probs, 1)

# ----------------------------
# Freshness Score (0–10)
# ----------------------------
conf = confidence.item()
pred_class = pred.item()

fresh_p = probs[0][0].item()
ripe_p = probs[0][1].item()
overripe_p = probs[0][2].item()

score = (
    fresh_p * 9 +
    ripe_p * 6 +
    overripe_p * 2
)

# ----------------------------
# Output
# ----------------------------
print("\n Class Probabilities:")
print(f"Fresh     : {probs[0][0]:.2f}")
print(f"Ripe      : {probs[0][1]:.2f}")
print(f"Overripe  : {probs[0][2]:.2f}")

print("\n Prediction:")
print(f"Class      → {classes[pred_class]}")
print(f"Confidence → {conf:.2f}")

print("\n Freshness Score (0–10):")
print(f"Score → {score:.2f}")
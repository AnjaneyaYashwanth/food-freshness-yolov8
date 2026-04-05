import streamlit as st
import torch
import cv2
import numpy as np
import torch.nn as nn
from skimage.feature import local_binary_pattern

from models.dual_stream_model import DualStreamModel
from ultralytics import YOLO

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Food Freshness Detection", layout="centered")

classes = ["fresh", "ripe", "overripe"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Paths
# ----------------------------
DUAL_MODEL_PATH = "best_dual_model.pth"
YOLO_BASE = "yolov8n-cls.pt"
YOLO_5CH_PATH = "yolo_5ch_food_optimized.pth"

# ----------------------------
# Load Dual Model
# ----------------------------
@st.cache_resource
def load_dual_model():
    model = DualStreamModel(num_classes=3, dropout=0.285)
    model.load_state_dict(torch.load(DUAL_MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# ----------------------------
# Load 5-Channel Model (FIXED)
# ----------------------------
@st.cache_resource
def load_5ch_model():
    yolo = YOLO(YOLO_BASE)
    model = yolo.model

    # Modify first conv → 5 channels
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
        nn.init.kaiming_normal_(new_conv.weight[:, 3:])

    model.model[0].conv = new_conv

    # Modify classifier
    model.model[-1].linear = nn.Linear(
        model.model[-1].linear.in_features, 3
    )

    # Load trained weights
    model.load_state_dict(torch.load(YOLO_5CH_PATH, map_location=device))

    model.to(device)
    model.eval()
    return model

dual_model = load_dual_model()
model_5ch = load_5ch_model()

# ----------------------------
# UI
# ----------------------------
st.markdown("<h1 style='text-align:center;'>🍌 Food Freshness Detection</h1>", unsafe_allow_html=True)

model_choice = st.selectbox("Select Model", ["Dual Stream", "5-Channel"])

st.divider()

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Read image using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    rgb = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if rgb is None:
        st.error("❌ Could not read image")
        st.stop()

    display_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    st.image(display_img, width=500)

    if st.button("🔍 Predict"):

        rgb = cv2.resize(rgb, (640, 640))

        # Common preprocessing
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 100, 200)

        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        lbp = np.uint8((lbp / lbp.max()) * 255)

        # ----------------------------
        # DUAL STREAM
        # ----------------------------
        if model_choice == "Dual Stream":

            rgb_n = rgb / 255.0
            edge_n = edge / 255.0
            lbp_n = lbp / 255.0

            rgb_t = torch.tensor(rgb_n, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            edge_t = torch.tensor(edge_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            lbp_t = torch.tensor(lbp_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            rgb_t, edge_t, lbp_t = rgb_t.to(device), edge_t.to(device), lbp_t.to(device)

            with torch.no_grad():
                outputs = dual_model(rgb_t, edge_t, lbp_t)
                probs = torch.softmax(outputs / 1.2, dim=1)

        # ----------------------------
        # 5-CHANNEL MODEL
        # ----------------------------
        else:

            rgb_n = rgb / 255.0
            edge_n = edge / 255.0
            lbp_n = lbp / 255.0

            rgb_t = np.transpose(rgb_n, (2, 0, 1))
            edge_t = np.expand_dims(edge_n, axis=0)
            lbp_t = np.expand_dims(lbp_n, axis=0)

            image_5ch = np.concatenate([rgb_t, edge_t, lbp_t], axis=0)
            image_5ch = torch.tensor(image_5ch, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model_5ch(image_5ch)

                # 🔥 IMPORTANT FIX (tuple handling)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                probs = torch.softmax(outputs, dim=1)

        # ----------------------------
        # Results
        # ----------------------------
        confidence, pred = torch.max(probs, 1)

        fresh_p = probs[0][0].item()
        ripe_p = probs[0][1].item()
        overripe_p = probs[0][2].item()

        # Smart score
        if pred.item() == 0:
            score = 7 + fresh_p * 3
        elif pred.item() == 1:
            score = 4 + ripe_p * 3
        else:
            score = 0 + overripe_p * 3

        # ----------------------------
        # UI Output
        # ----------------------------
        st.divider()
        st.subheader("🔍 Result")

        st.write(f"**Model:** {model_choice}")
        st.write(f"**Class:** {classes[pred.item()].upper()}")
        st.write(f"**Confidence:** {confidence.item():.2f}")
        st.write(f"**Freshness Score:** {score:.2f} / 10")

        st.subheader("📊 Probabilities")
        st.write(f"Fresh: {fresh_p:.2f}")
        st.write(f"Ripe: {ripe_p:.2f}")
        st.write(f"Overripe: {overripe_p:.2f}")
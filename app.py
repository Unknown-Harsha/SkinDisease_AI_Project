import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
import requests
import os

# ------------------------
# MODEL DOWNLOAD SECTION
# ------------------------
MODEL_FILE = "vit_base_patch16_224_best.pth"
MODEL_URL = "https://github.com/Unknown-Harsha/SkinDisease_AI_Project/releases/download/v1.0/vit_base_patch16_224_best.pth"

def ensure_model_downloaded():
    if not os.path.exists(MODEL_FILE):
        st.write("📥 Downloading model file from GitHub Release...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_FILE, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("✅ Model downloaded successfully!")

ensure_model_downloaded()

# ------------------------
# MODEL LOADING SECTION
# ------------------------
@st.cache_resource
def load_model():
    model = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=15)

    checkpoint = torch.load(MODEL_FILE, map_location=torch.device('cpu'))

    # Handles multiple save formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint

    model.eval()
    return model

model = load_model()

# ------------------------
# PREDICTION SECTION
# ------------------------
def predict_image(image):
    transform = timm.data.transforms_factory.transforms_imagenet_eval()
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][pred].item()
    return pred.item(), confidence * 100

# ------------------------
# STREAMLIT UI
# ------------------------
st.title("🩺 Skin Disease Detection AI")
st.write("Upload an image to classify common skin diseases and conditions.")

st.write("📸 You can either upload an image or use your webcam for live detection.")

option = st.radio("Select input method:", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Use Camera":
    camera_image = st.camera_input("Capture a skin image using your webcam")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

if image is not None:
    st.image(image, caption="Captured Image", use_column_width=True)
    st.write("🔍 Analyzing... please wait.")
    label, confidence = predict_image(image)
    st.success(f"✅ Predicted Label ID: {label}")
    st.info(f"Confidence: {confidence:.2f}%")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Analyzing... please wait.")
    label, confidence = predict_image(image)

    st.success(f"✅ Predicted Label ID: {label}")
    st.info(f"Confidence: {confidence:.2f}%")


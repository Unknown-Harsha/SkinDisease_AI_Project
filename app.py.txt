# ------------------------------
# Skin Disease Detection App
# ------------------------------

import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import requests
import os

# ------------------------------
# 1️⃣ Model Download Section
# ------------------------------

MODEL_FILE = "vit_base_patch16_224_best.pth"
MODEL_URL = "https://github.com/Unknown-Harsha/SkinDisease_AI_Project/releases/download/v1.0/vit_base_patch16_224_best.pth"

def ensure_model_downloaded():
    """Download the model file once if it's not already present."""
    if not os.path.exists(MODEL_FILE):
        st.info("Downloading model... please wait ⏳")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("✅ Model downloaded successfully!")

# Run download check before loading model
ensure_model_downloaded()

# ------------------------------
# 2️⃣ Model Loading Section
# ------------------------------

@st.cache_resource
def load_model():
    """Load the trained ViT model."""
    model_name = "vit_base_patch16_224"
    num_classes = 15  # adjust this to your dataset’s total classes
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ------------------------------
# 3️⃣ Prediction Function
# ------------------------------

# Example class names (change these to your dataset’s class names)
CLASSES = [
    "Melanoma",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatitis",
    "Eczema",
    "Rosacea",
    "Psoriasis",
    "Ringworm",
    "Vitiligo",
    "Acne",
    "Wart",
    "Fungal Infection",
    "Allergic Rash",
    "Birthmark",
    "Pimple"
]

def predict_image(image):
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred_class = torch.max(probs, 0)
        return CLASSES[pred_class.item()], conf.item()

# ------------------------------
# 4️⃣ Streamlit UI
# ------------------------------

st.set_page_config(page_title="Skin Disease Detection", layout="wide")

st.title("🧠 AI-Powered Skin Disease Detection System")
st.write("Upload an image of your skin condition and let the AI model predict the possible disease.")

uploaded_file = st.file_uploader("📸 Upload a skin image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image... ⏳"):
        label, confidence = predict_image(image)

    st.success(f"### 🩺 Predicted Disease: **{label}**")
    st.info(f"Confidence Level: {confidence*100:.2f}%")

st.markdown("---")
st.markdown("© 2025 | Skin Disease Detection AI | Developed by Harsha (ECE)")

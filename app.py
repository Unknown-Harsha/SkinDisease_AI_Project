import streamlit as st
import torch, os, urllib.request
import timm
from PIL import Image
import torchvision.transforms as transforms

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Skin Disease Detection", layout="centered")
st.title("🌿 AI-Based Skin Disease Detection")
st.write("Upload or capture a skin image to detect possible diseases using AI.")

# 🔗 Model file from your GitHub Release
MODEL_URL = "https://github.com/Unknown-Harsha/SkinDisease_AI_Project/releases/download/v1.0/vit_base_patch16_224_best.pth"
MODEL_FILE = "vit_base_patch16_224_best.pth"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.info("⬇️ Downloading model file (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        st.success("✅ Model downloaded successfully!")

    model = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=15)
    checkpoint = torch.load(MODEL_FILE, map_location=torch.device("cpu"))

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint

    model.eval()
    return model


model = load_model()

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    confidence = torch.softmax(outputs, dim=1)[0][predicted.item()] * 100
    return predicted.item(), confidence


# ---------------- LABELS ----------------
LABELS = [
    "Acne","Eczema","Psoriasis","Ringworm","Rosacea",
    "Vitiligo","Warts","Melanoma","Basal Cell Carcinoma",
    "Seborrheic Keratosis","Contact Dermatitis","Lichen Planus",
    "Pityriasis Rosea","Scabies","Urticaria (Hives)"
]

# ---------------- MAIN UI ----------------
option = st.radio("Select Input Method:", ["📁 Upload Image", "📷 Use Camera"])
image = None

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a skin image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.success("✅ Image uploaded successfully!")

elif option == "📷 Use Camera":
    camera_image = st.camera_input("Capture a skin image using your webcam:")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.success("📸 Image captured successfully!")

# ---------------- PREDICTION ----------------
if image is None:
    st.warning("⚠️ Please upload or capture an image to proceed.")
else:
    st.image(image, caption="Input Image", use_column_width=True)
    with st.spinner("🔍 Analyzing... please wait..."):
        label, confidence = predict_image(image)
        predicted_name = LABELS[label] if label < len(LABELS) else "Unknown"
    st.success(f"✅ Predicted Disease: **{predicted_name}**")
    st.info(f"📊 Confidence: {confidence:.2f}%")
    st.markdown("---")
    st.caption("⚕️ This AI prediction is for educational purposes and not a substitute for professional medical advice.")

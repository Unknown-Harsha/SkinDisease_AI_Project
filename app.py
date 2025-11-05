import streamlit as st
import torch
from PIL import Image
import timm

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=15)
    checkpoint = torch.load("vit_base_patch16_224_best.pth", map_location=torch.device('cpu'))

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint

    model.eval()
    return model


model = load_model()

# --- PREDICTION FUNCTION ---
def predict_image(image):
    import torchvision.transforms as transforms
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


# --- LABELS ---
LABELS = [
    "Acne", "Eczema", "Psoriasis", "Ringworm", "Rosacea",
    "Vitiligo", "Warts", "Melanoma", "Basal Cell Carcinoma",
    "Seborrheic Keratosis", "Contact Dermatitis", "Lichen Planus",
    "Pityriasis Rosea", "Scabies", "Urticaria (Hives)"
]


# --- MAIN APP UI ---
st.title("🌿 AI-Based Skin Disease Detection (Camera + Upload)")
st.write("Detect skin diseases in real-time using AI")

option = st.radio("Select input method:", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.success("✅ File upload successful")

elif option == "Use Camera":
    camera_image = st.camera_input("📸 Capture a skin image using your webcam")
    if camera_image is not None:
        st.write("📸 Camera captured an image")  # Debug message
        image = Image.open(camera_image).convert("RGB")

# --- DEBUG + PREDICTION LOGIC ---
if image is None:
    st.warning("⚠️ No image captured or uploaded yet.")
else:
    st.image(image, caption="Captured Image", use_column_width=True)
    with st.spinner("🔍 Analyzing... please wait..."):
        label, confidence = predict_image(image)
        predicted_name = LABELS[label] if label < len(LABELS) else "Unknown"
    st.success(f"✅ Predicted Disease: {predicted_name}")
    st.info(f"Confidence: {confidence:.2f}%")

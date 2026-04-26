import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retinal Disease Classifier",
    page_icon="👁️",
    layout="wide"
)

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_COLS  = ['N', 'D', 'G', 'C', 'A', 'H', 'M']
LABEL_NAMES = {
    'N': 'Normal',
    'D': 'Diabetes',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-related Macular Degeneration',
    'H': 'Hypertension',
    'M': 'Myopia'
}
MODEL_PATH = 'best_model.pth'
IMG_SIZE   = 224
THRESHOLD  = 0.5

# ── Preprocessing ──────────────────────────────────────────────────────────────
def apply_clahe(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    green = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img[:, :, 1] = clahe.apply(green)
    return img

def crop_fundus(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return image[y:y+h, x:x+w]

def preprocess(pil_image: Image.Image) -> np.ndarray:
    img = np.array(pil_image.convert('RGB'))
    img = crop_fundus(img)
    img = apply_clahe(img)
    return img

# ── Model ──────────────────────────────────────────────────────────────────────
class OdirModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=False)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = OdirModel(num_classes=7).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

# ── Transforms ─────────────────────────────────────────────────────────────────
from torchvision import transforms
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Grad-CAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer = model.backbone.blocks[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_gradcam(original_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    h, w    = original_rgb.shape[:2]
    cam_r   = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (0.6 * original_rgb + 0.4 * heatmap).astype(np.uint8)

# ── Inference ──────────────────────────────────────────────────────────────────
def predict(pil_image: Image.Image, model, device):
    processed = preprocess(pil_image)                        # CLAHE + crop
    pil_proc  = Image.fromarray(processed)
    tensor    = val_transform(pil_proc).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    gradcam   = GradCAM(model)
    with torch.enable_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).detach().cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    cam        = gradcam.generate(tensor, pred_class)
    overlay    = overlay_gradcam(
        np.array(pil_proc.resize((IMG_SIZE, IMG_SIZE))), cam
    )
    return probs, pred_class, processed, overlay

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

        .hero-title {
            font-family: 'Space Mono', monospace;
            font-size: 2.4rem;
            font-weight: 700;
            letter-spacing: -1px;
            color: #0f172a;
            margin-bottom: 0;
        }
        .hero-sub {
            font-size: 1rem;
            color: #64748b;
            margin-top: 4px;
            font-weight: 300;
        }
        .result-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 0.8rem;
        }
        .disease-tag {
            display: inline-block;
            background: #fee2e2;
            color: #991b1b;
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .normal-tag {
            background: #dcfce7;
            color: #166534;
        }
        .metric-label {
            font-size: 0.78rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-family: 'Space Mono', monospace;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 600;
            color: #0f172a;
            font-family: 'Space Mono', monospace;
        }
        .section-header {
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #94a3b8;
            margin-bottom: 0.6rem;
        }
        div[data-testid="stFileUploader"] {
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="hero-title">👁️ Retinal Disease Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a fundus image — EfficientNet-B3 + Grad-CAM · Trained on ODIR-5K · Macro AUC 0.8623</p>', unsafe_allow_html=True)
st.divider()

# Sidebar — info
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This tool classifies **7 retinal conditions** from fundus photographs:

    | Code | Condition |
    |------|-----------|
    | N | Normal |
    | D | Diabetic Retinopathy |
    | G | Glaucoma |
    | C | Cataract |
    | A | AMD |
    | H | Hypertension |
    | M | Myopia |

    **Model:** EfficientNet-B3 (transfer learning)  
    **Dataset:** ODIR-5K (2,949 patients)  
    **Preprocessing:** CLAHE + Fundus Crop  
    **Best Macro AUC:** 0.8623  

    ---
    ⚠️ *For research purposes only. Not a medical device.*
    """)

# Load model
try:
    model, device = load_model()
    st.sidebar.success(f"Model loaded · {str(device).upper()}")
except Exception as e:
    st.error(f"Could not load model: {e}. Make sure `best_model.pth` is in the same folder as `app.py`.")
    st.stop()

# Upload
uploaded = st.file_uploader(
    "Upload a fundus image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    pil_image = Image.open(uploaded).convert('RGB')

    with st.spinner("Running inference + Grad-CAM..."):
        probs, pred_class, processed, overlay = predict(pil_image, model, device)

    # ── Results layout ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown('<p class="section-header">Original Image</p>', unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Preprocessed (CLAHE + Crop)</p>', unsafe_allow_html=True)
        st.image(processed, use_container_width=True)

    with col3:
        st.markdown('<p class="section-header">Grad-CAM Overlay</p>', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)

    st.divider()

    # ── Probabilities ──────────────────────────────────────────────────────────
    res_col, chart_col = st.columns([1, 1.4])

    with res_col:
        st.markdown('<p class="section-header">Predictions</p>', unsafe_allow_html=True)

        detected = [LABEL_COLS[i] for i, p in enumerate(probs) if p >= THRESHOLD]

        if 'N' in detected or not detected:
            st.markdown(f'<span class="disease-tag normal-tag">✓ Normal</span>', unsafe_allow_html=True)
        else:
            for code in detected:
                if code != 'N':
                    st.markdown(f'<span class="disease-tag">⚠ {LABEL_NAMES[code]}</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        for i, (code, prob) in enumerate(zip(LABEL_COLS, probs)):
            bar_color = "#ef4444" if prob >= THRESHOLD and code != 'N' else "#22c55e" if code == 'N' and prob >= THRESHOLD else "#94a3b8"
            st.markdown(f"""
                <div style="margin-bottom:10px">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                        <span style="font-size:0.85rem;font-weight:600;color:#0f172a">{LABEL_NAMES[code]}</span>
                        <span style="font-size:0.85rem;font-family:'Space Mono',monospace;color:#64748b">{prob:.1%}</span>
                    </div>
                    <div style="background:#e2e8f0;border-radius:4px;height:8px">
                        <div style="background:{bar_color};width:{prob*100:.1f}%;height:8px;border-radius:4px;transition:width 0.3s"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    with chart_col:
        st.markdown('<p class="section-header">Confidence Chart</p>', unsafe_allow_html=True)
        import pandas as pd
        chart_data = pd.DataFrame({
            'Disease': [LABEL_NAMES[c] for c in LABEL_COLS],
            'Probability': probs
        }).set_index('Disease')
        st.bar_chart(chart_data, height=320)

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.caption("⚠️ This tool is for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.")

else:
    # Placeholder when no image uploaded
    st.info("👆 Upload a retinal fundus image to get started.")
    st.markdown("""
    **What this app does:**
    1. Preprocesses your fundus image with CLAHE contrast enhancement and fundus circle cropping
    2. Runs it through a fine-tuned EfficientNet-B3 model
    3. Shows predicted probabilities for 7 retinal conditions
    4. Generates a Grad-CAM heatmap showing which regions influenced the prediction
    """)

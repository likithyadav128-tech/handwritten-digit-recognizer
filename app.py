"""
app.py
------
Streamlit app for handwritten digit recognition. Draw a digit directly
on the canvas, or upload a photo of one — either way it's reduced to a
28x28 grid and classified by a model trained on MNIST.

Run locally:
    streamlit run app.py

Deploy: push this file + requirements.txt + model/ to GitHub, then on
share.streamlit.io point the "Main file path" to app.py.
"""

import numpy as np
import joblib
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

MODEL_PATH = "model/digit_classifier.joblib"
TARGET_SIZE = 28
MARGIN_FRAC = 0.18

st.set_page_config(page_title="Digit Reader", page_icon="⠿", layout="centered")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["model_name"], bundle["test_accuracy"]


def binary_to_features(binary, margin_frac=MARGIN_FRAC, target_size=TARGET_SIZE):
    """Take a binarized (0/1) digit array, crop/pad/downsample it into a
    28x28, 0-1 scaled feature vector matching MNIST's format."""

    rows = np.any(binary > 0, axis=1)
    cols = np.any(binary > 0, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError("No digit detected — draw or upload a clearer digit.")
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = binary[rmin:rmax + 1, cmin:cmax + 1]

    # Pad proportional to the digit's own size so it fills roughly the same
    # fraction of the frame as real MNIST digits (not edge-to-edge).
    h, w = cropped.shape
    pad = int(round(max(h, w) * margin_frac))
    size = max(h, w) + 2 * pad
    square = np.zeros((size, size), dtype=np.float64)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped

    # Box-average downsample: each output cell becomes the fraction of "on"
    # pixels in the corresponding source block — a density count, similar to
    # how MNIST digits were anti-aliased. Far less sensitive to stroke
    # thickness than a smoothing resize.
    img_bin = Image.fromarray((square * 255).astype(np.uint8))
    img_small = img_bin.resize((target_size, target_size), Image.BOX)
    arr_small = np.array(img_small).astype(np.float64) / 255.0

    return arr_small.flatten().reshape(1, -1)


def photo_to_features(pil_img):
    """Convert an uploaded photo of a digit into model-ready features."""
    img = pil_img.convert("L")  # grayscale
    arr = np.array(img).astype(np.float64)

    # Auto-detect polarity: training data is bright digit on dark background.
    if arr.mean() > 127:
        arr = 255.0 - arr

    thresh = arr.max() * 0.15
    binary = (arr > thresh).astype(np.float64)

    return binary_to_features(binary)


def canvas_to_features(canvas_image):
    """Convert the drawable canvas's RGBA array into model-ready features."""
    # The canvas's alpha channel is opaque everywhere (including the plain
    # black background), so it can't tell us where you actually drew.
    # Detect strokes by brightness instead: strokes are white, background
    # is black.
    rgb = canvas_image[:, :, :3].astype(np.float64)
    gray = rgb.mean(axis=2)
    binary = (gray > 127).astype(np.float64)
    return binary_to_features(binary)


# ---- chalkboard-styled CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #1E2B25; color: #F2EFE6; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.digit-box {
    font-family: 'Space Mono', monospace;
    font-size: 5rem;
    font-weight: 700;
    text-align: center;
    color: #F2EFE6;
    background: #26362F;
    border: 1px solid rgba(242,239,230,0.1);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 10px;
}
.chip {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #9FB3A8;
    border: 1px solid rgba(242,239,230,0.14);
    border-radius: 20px;
    padding: 4px 14px;
    margin-right: 6px;
    display: inline-block;
}
.chip.top { color: #1E2B25; background: #E8C468; border-color: #E8C468; }
.model-tag { color: #9FB3A8; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)

model, model_name, test_acc = load_model()

st.markdown(f"### ⠿ DIGIT READER")
st.markdown(f"<span class='model-tag'>{model_name} · {test_acc:.1%} accuracy</span>", unsafe_allow_html=True)
st.markdown("#### Draw a number, or upload one. It reads it back.")
st.write(
    "Every input gets reduced to a 28×28 grid — the same resolution the "
    "model was trained on. That grid, not the drawing or photo, is what "
    "decides the answer."
)

mode = st.radio("Input method", ["Draw", "Upload photo"], horizontal=True, label_visibility="collapsed")

features = None
display_col_left = None

if mode == "Draw":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.caption("draw here")
        canvas_result = st_canvas(
            fill_color="#FFFFFF",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    if canvas_result.image_data is not None and canvas_result.image_data[:, :, :3].mean(axis=2).max() > 127:
        try:
            features = canvas_to_features(canvas_result.image_data)
        except ValueError:
            features = None

else:
    uploaded = st.file_uploader("Drop a digit image here", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        col1, col2 = st.columns([1, 1])
        pil_img = Image.open(uploaded)
        with col1:
            st.image(pil_img, caption="your upload", use_container_width=True)
        try:
            features = photo_to_features(pil_img)
        except ValueError as e:
            st.error(str(e))

if features is not None:
    with col2:
        grid = features.reshape(TARGET_SIZE, TARGET_SIZE)
        grid_img = Image.fromarray((grid * 255).astype(np.uint8)).resize((160, 160), Image.NEAREST)
        st.image(grid_img, caption=f"what the model sees ({TARGET_SIZE}×{TARGET_SIZE})", width=160)

    pred = int(model.predict(features)[0])

    st.markdown(f"<div class='digit-box'>{pred}</div>", unsafe_allow_html=True)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = proba[pred]
        st.progress(float(confidence))
        st.write(f"**Confidence:** {confidence:.2%}")

        top3_idx = np.argsort(proba)[::-1][:3]
        chips_html = ""
        for i, d in enumerate(top3_idx):
            cls = "chip top" if i == 0 else "chip"
            chips_html += f"<span class='{cls}'>{d} · {proba[d]:.1%}</span>"
        st.markdown(chips_html, unsafe_allow_html=True)

st.markdown("---")
st.caption("Trained on the MNIST handwritten digit dataset. No image is stored — everything runs in memory.")
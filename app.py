"""
app.py
------
Streamlit app for handwritten digit recognition.

Run locally:
    streamlit run app.py

Deploy: push this file + requirements.txt + model/ to GitHub, then on
share.streamlit.io point the "Main file path" to app.py.
"""

import numpy as np
import joblib
from PIL import Image
import streamlit as st

MODEL_PATH = "model/digit_classifier.joblib"

st.set_page_config(page_title="Digit Reader", page_icon="⠿", layout="centered")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["model_name"], bundle["test_accuracy"]


def preprocess_image(pil_img, margin_frac=0.18, target_size=28):
    """Convert an arbitrary PIL image of a single handwritten digit into a
    28x28, 0-1 scaled array matching MNIST's format."""

    img = pil_img.convert("L")  # grayscale
    arr = np.array(img).astype(np.float64)

    # Auto-detect polarity: training data is bright digit on dark background.
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Threshold and binarize. sklearn's load_digits images were built by
    # binarizing a NIST bitmap and then counting the fraction of "on" pixels
    # per 4x4 block to get each 0-16 cell value — a density count, not a
    # smoothed grayscale resize. Binarizing here (instead of keeping
    # continuous gray values) lets the later box-average resize reproduce
    # that density count faithfully, and makes the result far less sensitive
    # to stroke thickness (marker vs. pen).
    thresh = arr.max() * 0.15
    binary = (arr > thresh).astype(np.float64)

    # Crop tightly to the digit's bounding box
    rows = np.any(binary > 0, axis=1)
    cols = np.any(binary > 0, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError("No digit detected in image — try a clearer photo with more contrast.")
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = binary[rmin:rmax + 1, cmin:cmax + 1]

    # Pad proportional to the digit's own size (not a fixed pixel count).
    # sklearn's load_digits images sit with visible margin — the digit fills
    # roughly 70-80% of the 8x8 frame, not edge-to-edge. A fixed 1px pad on a
    # high-res photo crop was effectively zero margin, so the digit filled
    # ~99% of the frame and got distorted at 8x8 in a way the model never saw
    # in training (this is what was turning "4" into "3").
    h, w = cropped.shape
    pad = int(round(max(h, w) * margin_frac))
    size = max(h, w) + 2 * pad
    square = np.zeros((size, size), dtype=np.float64)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped

    # Box-average downsample: each output cell becomes the fraction of "on"
    # pixels in the corresponding source block — a density count, similar to
    # how MNIST digits were anti-aliased/normalized. (LANCZOS instead smooths
    # edges together, which is what let thick strokes blur into other digits.)
    img_bin = Image.fromarray((square * 255).astype(np.uint8))
    imgN = img_bin.resize((target_size, target_size), Image.BOX)
    arrN = np.array(imgN).astype(np.float64) / 255.0

    return arrN.flatten().reshape(1, -1)


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
st.markdown("#### Show it a number. It reads it back.")
st.write(
    "Every image gets reduced to a 28×28 grid — the same resolution the "
    "model was trained on. That grid, not the photo, is what decides the "
    "answer."
)

uploaded = st.file_uploader("Drop a digit image here", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    col1, col2 = st.columns([1, 1])
    pil_img = Image.open(uploaded)

    with col1:
        st.image(pil_img, caption="your upload", use_container_width=True)

    try:
        features = preprocess_image(pil_img)

        with col2:
            grid = features.reshape(28, 28)
            grid_img = Image.fromarray((grid * 255).astype(np.uint8)).resize((160, 160), Image.NEAREST)
            st.image(grid_img, caption="what the model sees (28×28)", width=160)

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

    except ValueError as e:
        st.error(str(e))

st.markdown("---")
st.caption("Trained on the MNIST handwritten digit dataset. No image is stored — everything runs in memory.")
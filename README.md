# Digit Reader

Draw a digit on the canvas, or upload a photo of one — it reads it back.

Every input gets reduced to a 28×28 grid — the same resolution the model
was trained on (MNIST). The classifier is a small MLP (scikit-learn
`MLPClassifier`, hidden layers 128→64) trained on all ~70,000 MNIST samples.

## Project structure

```
app.py             Streamlit app (draw or upload -> preprocess -> predict)
train_model.py      One-time training script, produces model/digit_classifier.joblib
model/
  digit_classifier.joblib   Trained model bundle (model, name, test accuracy)
requirements.txt
```

## Setup

```
pip install -r requirements.txt
python train_model.py     # downloads MNIST (first run only) and trains
streamlit run app.py
```

## Deploy

Push this repo to GitHub (including the trained `model/digit_classifier.joblib`),
then on [share.streamlit.io](https://share.streamlit.io) point the
"Main file path" to `app.py`.

## How the preprocessing works

Both the canvas drawing and uploaded photos go through the same pipeline
(`binary_to_features` in `app.py`):

1. Convert to a binary (0/1) digit mask — for photos, grayscale + auto
   polarity detection + threshold; for the canvas, the drawn stroke's
   alpha channel.
2. Crop tightly to the digit's bounding box.
3. Pad proportionally to the digit's own size (not a fixed pixel count),
   so it fills roughly the same fraction of the frame as real MNIST digits.
4. Downsample with a box-average filter (density count per block) rather
   than a smoothing resize — this is much less sensitive to stroke
   thickness (marker vs. pen vs. mouse) than a naive resize.

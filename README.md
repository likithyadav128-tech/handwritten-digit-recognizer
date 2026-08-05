# Digit Reader — Streamlit version

This is the Streamlit-native version of the app, built for deploying on
**Streamlit Community Cloud** (the earlier Flask version can't run there —
Streamlit Cloud runs `streamlit run app.py`, not a Flask server).

## Folder structure

```
streamlit_app/
├── app.py                          # Streamlit app (UI + prediction logic)
├── requirements.txt
└── model/
    └── digit_classifier.joblib     # trained model (already included)
```

## Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

It opens automatically at `http://localhost:8501`.

## Fixing your existing Streamlit Cloud deployment

Your app crashed because the repo's `app.py` was the **Flask** version
(it called `app.run(...)`, which only works with a real Flask server, not
inside Streamlit's runtime). To fix it:

1. In your GitHub repo (`handwritten-digit-recognizer`), **replace** the
   contents of `app.py` with the `app.py` from this folder.
2. Also replace `requirements.txt` with the one from this folder — it drops
   Flask/gunicorn, which aren't needed anymore.
3. Make sure `model/digit_classifier.joblib` is still in the repo at
   `model/digit_classifier.joblib` (same path as before).
4. Commit and push. Streamlit Cloud will auto-redeploy in a minute or two.

If it still fails, on the app page click **Manage app** (bottom right) →
check the logs for the exact error.

## Notes

- Images are processed in memory only — nothing is saved to disk.
- `@st.cache_resource` keeps the model loaded in memory across requests
  instead of reloading it on every upload.

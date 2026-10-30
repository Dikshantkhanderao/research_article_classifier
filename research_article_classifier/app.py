import html
import json
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "best_model.joblib"
VECTORIZER_PATH = BASE_DIR / "artifacts" / "tfidf_vectorizer.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "model_metadata.json"

CUSTOM_STOP_WORDS = {
    "background", "methods", "method", "results", "result", "conclusion",
    "conclusions", "objective", "objectives", "purpose", "introduction",
    "elsevier", "copyright", "rights", "reserved", "ltd", "inc", "wiley",
    "springer", "emerald", "taylor", "francis", "sage", "publisher", "published",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | CUSTOM_STOP_WORDS


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, vectorizer, metadata


def preprocess_text(raw_text):
    """Keep this preprocessing identical to the training script."""
    text = html.unescape(str(raw_text)).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [
        token for token in tokens
        if token not in STOP_WORDS and len(token) > 2 and not token.isdigit()
    ]
    return " ".join(tokens)


st.set_page_config(
    page_title="Research Article Classifier",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Article Classification")
st.write(
    "Upload a CSV file containing research-paper abstracts. "
    "The model predicts whether each abstract belongs to Science, Business, or Social Science."
)

try:
    model, vectorizer, metadata = load_artifacts()
except Exception as exc:
    st.error("Could not load the trained model. Check that the artifacts folder is present.")
    st.exception(exc)
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Prediction categories", str(len(metadata.get("classes", []))))
col2.metric("Selected model", metadata.get("selected_model", "Saved classifier"))
col3.metric("Test accuracy", f"{metadata.get('test_accuracy', 0) * 100:.2f}%")

st.info("Your CSV must contain a column named **Abstract**. Column names are not case-sensitive.")

uploaded_file = st.file_uploader("Upload your abstracts CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        abstract_col = next(
            (col for col in input_df.columns if str(col).strip().lower() == "abstract"),
            None,
        )

        if abstract_col is None:
            st.error("The uploaded CSV does not contain an 'Abstract' column.")
            st.write("Available columns:", list(input_df.columns))
        else:
            result_df = input_df.copy()
            cleaned = result_df[abstract_col].fillna("").astype(str).apply(preprocess_text)
            valid_mask = cleaned.str.strip().ne("")

            result_df["Predicted Category"] = "Not enough text"
            result_df["Prediction Confidence (%)"] = pd.NA

            if valid_mask.any():
                features = vectorizer.transform(cleaned[valid_mask])
                predictions = model.predict(features)
                result_df.loc[valid_mask, "Predicted Category"] = predictions

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)
                    confidence = probabilities.max(axis=1) * 100
                    result_df.loc[valid_mask, "Prediction Confidence (%)"] = confidence.round(2)

            st.success(f"Finished! Processed {len(result_df)} rows.")
            st.subheader("Prediction results")
            st.dataframe(result_df, use_container_width=True)

            predicted_counts = (
                result_df.loc[
                    result_df["Predicted Category"] != "Not enough text",
                    "Predicted Category",
                ]
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Number of abstracts")
            )

            if not predicted_counts.empty:
                left, right = st.columns(2)
                with left:
                    st.subheader("Predicted category counts")
                    st.bar_chart(predicted_counts.set_index("Category"))
                with right:
                    st.subheader("Count summary")
                    st.dataframe(predicted_counts, use_container_width=True)

            output = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download predictions as CSV",
                data=output,
                file_name="classified_research_abstracts.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error("There was a problem reading or classifying this CSV file.")
        st.exception(exc)

st.divider()
st.caption(
    "Academic project baseline. Predictions are model estimates and should be reviewed "
    "before using them as formal publication classifications."
)

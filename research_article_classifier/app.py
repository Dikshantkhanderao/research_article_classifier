import html
import json
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "artifacts" / "best_model.joblib"
VECTORIZER_PATH = BASE_DIR / "artifacts" / "tfidf_vectorizer.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "model_metadata.json"


# ==========================================
# 2. TEXT PREPROCESSING
# ==========================================

CUSTOM_STOP_WORDS = {
    "background", "methods", "method", "results",
    "result", "conclusion", "conclusions",
    "objective", "objectives", "purpose",
    "introduction", "elsevier", "copyright",
    "rights", "reserved", "ltd", "inc",
    "wiley", "springer", "emerald",
    "taylor", "francis", "sage",
    "publisher", "published"
}

STOP_WORDS = set(ENGLISH_STOP_WORDS) | CUSTOM_STOP_WORDS


def preprocess_text(raw_text):

    text = html.unescape(str(raw_text)).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    tokens = [
        token for token in tokens
        if token not in STOP_WORDS
        and len(token) > 2
        and not token.isdigit()
    ]

    return " ".join(tokens)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(VECTORIZER_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, vectorizer, metadata


# ==========================================
# 4. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Research Article Classifier",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Article Classification")

st.write(
    "Enter a research abstract manually or upload a CSV file. "
    "The model predicts whether the abstract belongs to "
    "Science, Business, or Social Science."
)


# ==========================================
# 5. LOAD MODEL SAFELY
# ==========================================

try:

    model, vectorizer, metadata = load_artifacts()

except Exception as exc:

    st.error(
        "Could not load the trained model. "
        "Please check the artifacts folder and model files."
    )

    st.exception(exc)

    st.stop()


# ==========================================
# 6. MODEL INFORMATION
# ==========================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Prediction Categories",
    str(len(metadata.get("classes", [])))
)

col2.metric(
    "Selected Model",
    metadata.get("selected_model", "Saved Classifier")
)

col3.metric(
    "Test Accuracy",
    f"{metadata.get('test_accuracy', 0) * 100:.2f}%"
)


# ==========================================
# 7. PREDICTION INTERFACE
# ==========================================

st.divider()

st.subheader("🔍 Predict Research Article Category")

manual_tab, csv_tab = st.tabs(
    ["✍️ Single Abstract Prediction", "📁 CSV File Prediction"]
)


# ==========================================
# 8. SINGLE ABSTRACT PREDICTION
# ==========================================

with manual_tab:

    st.write(
        "Enter or paste a research paper abstract "
        "to predict its publication category."
    )

    abstract_text = st.text_area(
        "Enter Research Abstract",
        placeholder="Paste your research abstract here...",
        height=220
    )

    if st.button(
        "Predict Category",
        type="primary",
        key="single_prediction"
    ):

        if not abstract_text.strip():

            st.warning("Please enter a research abstract.")

        else:

            cleaned_abstract = preprocess_text(abstract_text)

            if len(cleaned_abstract.split()) < 3:

                st.warning(
                    "Please enter a longer abstract "
                    "containing meaningful research text."
                )

            else:

                features = vectorizer.transform(
                    [cleaned_abstract]
                )

                prediction = model.predict(features)[0]

                st.success(
                    f"Predicted Category: {prediction}"
                )

                # Prediction probabilities

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(features)[0]

                    classes = model.classes_

                    confidence = float(
                        probabilities.max() * 100
                    )

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

                    probability_df = pd.DataFrame({
                        "Category": classes,
                        "Probability (%)": (
                            probabilities * 100
                        ).round(2)
                    })

                    probability_df = probability_df.sort_values(
                        "Probability (%)",
                        ascending=False
                    )

                    st.subheader("Category Probability")

                    st.bar_chart(
                        probability_df.set_index("Category")
                    )

                    st.dataframe(
                        probability_df,
                        hide_index=True,
                        use_container_width=True
                    )

                    st.caption(
                        "Model probabilities are estimates, "
                        "not guarantees of correctness."
                    )

                else:

                    st.info(
                        "This model does not provide probability estimates."
                    )


# ==========================================
# 9. CSV FILE PREDICTION
# ==========================================

with csv_tab:

    st.write(
        "Upload a CSV file containing multiple research abstracts."
    )

    st.info(
        "Your CSV must contain a column named Abstract. "
        "Column names are not case-sensitive."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        key="csv_prediction"
    )

    if uploaded_file is not None:

        try:

            input_df = pd.read_csv(uploaded_file)

            abstract_col = next(
                (
                    col for col in input_df.columns
                    if str(col).strip().lower() == "abstract"
                ),
                None
            )

            if abstract_col is None:

                st.error(
                    "CSV file must contain an Abstract column."
                )

                st.write(
                    "Available columns:",
                    list(input_df.columns)
                )

            else:

                result_df = input_df.copy()

                cleaned = (
                    result_df[abstract_col]
                    .fillna("")
                    .astype(str)
                    .apply(preprocess_text)
                )

                valid_mask = cleaned.str.strip().ne("")

                result_df["Predicted Category"] = "Not enough text"

                result_df["Prediction Confidence (%)"] = pd.NA

                if valid_mask.any():

                    features = vectorizer.transform(
                        cleaned[valid_mask]
                    )

                    predictions = model.predict(features)

                    result_df.loc[
                        valid_mask,
                        "Predicted Category"
                    ] = predictions

                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(features)

                        confidence = (
                            probabilities.max(axis=1) * 100
                        )

                        result_df.loc[
                            valid_mask,
                            "Prediction Confidence (%)"
                        ] = confidence.round(2)

                # Display results

                st.success(
                    f"Finished! Processed {len(result_df)} rows."
                )

                st.subheader("📊 Prediction Results")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                # Category counts

                predicted_counts = (
                    result_df.loc[
                        result_df["Predicted Category"]
                        != "Not enough text",
                        "Predicted Category"
                    ]
                    .value_counts()
                    .rename_axis("Category")
                    .reset_index(name="Number of Abstracts")
                )

                if not predicted_counts.empty:

                    left, right = st.columns(2)

                    with left:

                        st.subheader(
                            "Predicted Category Counts"
                        )

                        st.bar_chart(
                            predicted_counts.set_index("Category")
                        )

                    with right:

                        st.subheader("Count Summary")

                        st.dataframe(
                            predicted_counts,
                            use_container_width=True
                        )

                # Download predictions

                output = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=output,
                    file_name="classified_research_abstracts.csv",
                    mime="text/csv"
                )

        except Exception as exc:

            st.error(
                "There was a problem reading or classifying this CSV file."
            )

            st.exception(exc)


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.caption(
    "Research Article Classification | Machine Learning Project. "
    "Predictions are model estimates and should be reviewed "
    "before formal publication classification."
)

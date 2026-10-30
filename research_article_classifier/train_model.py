"""Train and save the six research-abstract classifiers used by the Streamlit demo."""
import html
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "research_abstracts_labeled.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

CUSTOM_STOP_WORDS = {
    "background", "methods", "method", "results", "result", "conclusion",
    "conclusions", "objective", "objectives", "purpose", "introduction",
    "elsevier", "copyright", "rights", "reserved", "ltd", "inc", "wiley",
    "springer", "emerald", "taylor", "francis", "sage", "publisher", "published",
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
        if token not in STOP_WORDS and len(token) > 2 and not token.isdigit()
    ]
    return " ".join(tokens)


def main():
    data = pd.read_csv(DATA_PATH)
    if not {"Abstract", "Label"}.issubset(data.columns):
        raise ValueError("Dataset must contain 'Abstract' and 'Label' columns.")

    data = data.dropna(subset=["Abstract", "Label"]).copy()
    data["Abstract_nlp"] = data["Abstract"].astype(str).apply(preprocess_text)
    data = data[data["Abstract_nlp"].str.strip().ne("")]

    X = data["Abstract_nlp"]
    y = data["Label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=100, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "kNN": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
        "SVM": LinearSVC(random_state=42, max_iter=5000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        predictions = model.predict(X_test_vec)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision_macro": precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "Recall_macro": recall_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "F1_macro": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
        })

    metrics = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    best_name = metrics.iloc[0]["Model"]
    best_model = models[best_name]

    joblib.dump(best_model, ARTIFACT_DIR / "best_model.joblib")
    joblib.dump(vectorizer, ARTIFACT_DIR / "tfidf_vectorizer.joblib")
    metrics.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    metadata = {
        "selected_model": best_name,
        "test_accuracy": float(metrics.iloc[0]["Accuracy"]),
        "classes": sorted(y.unique().tolist()),
        "total_records": int(len(data)),
        "training_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "random_state": 42,
        "preprocessing": "HTML decode, lowercase, URL/email/tag removal, punctuation cleanup, scikit-learn English stopwords plus custom stopwords, remove tokens shorter than 3 and numeric-only tokens; no lemmatization.",
    }
    with open(ARTIFACT_DIR / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(metrics.to_string(index=False))
    print(f"\\nSaved best model: {best_name}")


if __name__ == "__main__":
    main()

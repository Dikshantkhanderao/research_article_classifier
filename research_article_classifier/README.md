# Research Article Classification

A Streamlit demo that classifies research-paper abstracts into **Science**, **Business**, or **Social Science**.

## Project structure

```text
research_article_classifier/
├── app.py
├── requirements.txt
├── train_model.py
├── README.md
├── artifacts/
│   ├── best_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── model_metadata.json
│   └── model_comparison.csv
└── data/
    ├── Science.csv
    ├── Business.csv
    ├── social_science.csv
    └── research_abstracts_labeled.csv
```

## Retrain the models

To train all six classifiers again from the combined labelled dataset and regenerate the files in `artifacts/`, run:

```bash
python train_model.py
```

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed in the terminal (usually `http://localhost:8501`).
6. Upload a CSV file with a column named `Abstract`.
7. Review the predicted categories and download the result CSV.

## Deploy using GitHub and Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the complete contents of this folder to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose your repository and branch.
5. Set the main file path to `app.py`.
6. Click **Deploy**.

Keep `app.py`, `requirements.txt`, and the `artifacts/` folder in the repository. The app needs both `.joblib` files to make predictions.

## Input CSV format

The uploaded file must contain an `Abstract` column. Other columns are allowed and are preserved in the output.

Example:

| Title | Abstract |
|---|---|
| Example paper | This paper studies ... |

The app adds:
- `Predicted Category`
- `Prediction Confidence (%)` (when supported by the saved model)

Rows with blank abstracts are marked `Not enough text`.

## Dataset

The three supplied Scopus-export CSV files were combined and labelled. After removing rows with missing title/abstract and removing duplicate abstracts (including abstracts that had conflicting labels), the prepared dataset contains **1,472 records**:

- Science: 497
- Business: 489
- Social Science: 486

The original three CSV files are retained in `data/`. The combined labelled dataset is `data/research_abstracts_labeled.csv`.

## Models compared

The project compared:
- Logistic Regression
- Decision Tree
- Naive Bayes
- k-Nearest Neighbors
- Support Vector Machine (Linear SVM)
- Random Forest

The saved model is the model with the highest accuracy on the held-out test split. In this run, **Random Forest** was selected.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 96.27% | 96.28% |
| Naive Bayes | 93.90% | 93.93% |
| SVM | 93.90% | 93.91% |
| Decision Tree | 93.56% | 93.58% |
| Logistic Regression | 93.22% | 93.21% |
| kNN | 83.05% | 82.93% |

The split used 80% training data and 20% test data, with `random_state=42` and stratification by class. These are results on this particular held-out split, not a guarantee of future performance.

## Preprocessing note

The deployment artifacts were trained with the portable preprocessing function in `train_model.py` / `app.py`: HTML decoding, lowercase conversion, URL/email/HTML-tag removal, punctuation cleanup, stopword removal, and removal of very short or numeric-only tokens.

The uploaded notebook uses NLTK stopwords and WordNet lemmatization. The deployment bundle uses scikit-learn's built-in English stopword list and does not use WordNet lemmatization, so the saved model and Streamlit app use exactly the same preprocessing without requiring a runtime NLTK corpus download. If you retrain the model, use the included training script and keep its preprocessing unchanged.

## Important

- Do not upload private or copyrighted datasets unless you have permission to use and redistribute them.
- The model is a baseline classifier. It can make mistakes, especially for interdisciplinary abstracts.
- Check the output before treating predictions as final labels.

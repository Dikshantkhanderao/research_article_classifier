except Exception as exc:
    st.error("Could not load the trained model. Check that the artifacts folder is present.")
    st.exception(exc)
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Prediction categories", str(len(metadata.get("classes", []))))
col2.metric("Selected model", metadata.get("selected_model", "Saved classifier"))
col3.metric("Test accuracy", f"{metadata.get('test_accuracy', 0) * 100:.2f}%")

st.subheader("Predict a research article category")
manual_tab, csv_tab = st.tabs(["✍️ Single abstract", "📁 Upload CSV"])

with manual_tab:
    st.write("Paste or type one research-paper abstract below to predict its field.")
    abstract_text = st.text_area(
        "Research abstract",
        placeholder="Paste the complete research abstract here...",
        height=220,
        key="single_abstract_input",
    )
    if st.button("Predict category", type="primary", key="predict_single"):
        cleaned_abstract = preprocess_text(abstract_text)
        if not abstract_text.strip():
            st.warning("Please enter a research abstract first.")
        elif len(cleaned_abstract.split()) < 3:
            st.warning("Please enter a longer abstract with meaningful research text.")
        else:
            features = vectorizer.transform([cleaned_abstract])
            prediction = model.predict(features)[0]
            st.success(f"Predicted category: **{prediction}**")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                classes = model.classes_
                confidence = float(probabilities.max() * 100)
                st.metric("Prediction confidence", f"{confidence:.2f}%")
                probability_df = pd.DataFrame({
                    "Category": classes,
                    "Model probability (%)": (probabilities * 100).round(2),
                }).sort_values("Model probability (%)", ascending=False)
                st.caption("Model probabilities are estimates, not guarantees of correctness.")
                st.dataframe(probability_df, hide_index=True, use_container_width=True)
            else:
                st.caption("This model does not provide probability estimates.")

with csv_tab:
    st.info("Your CSV must contain a column named **Abstract**. Column names are not case-sensitive.")
    uploaded_file = st.file_uploader("Upload your abstracts CSV file", type=["csv"], key="csv_upload")

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

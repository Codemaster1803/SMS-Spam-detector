st.markdown("---")
st.header("🔄 Admin: Retrain Model with Feedback")
if st.button("🚀 Retrain Now"):
    try:
        # 1️⃣ Load original dataset
        original = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
        original.columns = ['label', 'message']
        original['label'] = original['label'].map({'ham': 0, 'spam': 1})

        # 2️⃣ Load feedback and oversample it
        try:
            feedback = pd.read_csv("feedback.csv", names=['message', 'label'])
            feedback['label'] = feedback['label'].astype(int)

            # Oversample feedback x10 to make corrections more impactful
            feedback_oversampled = pd.concat([feedback]*10, ignore_index=True)

            combined = pd.concat([original, feedback_oversampled], ignore_index=True)
            st.info(f"✅ Loaded {len(feedback)} feedback entries (oversampled for stronger influence).")
        except FileNotFoundError:
            combined = original
            st.warning("⚠️ No feedback found. Training with original dataset only.")

        # 3️⃣ Preprocess
        st.write("🔄 Preprocessing messages...")
        combined['transformed'] = combined['message'].apply(transform_text)

        # 4️⃣ Vectorize
        st.write("🔠 Vectorizing with TF-IDF...")
        tfidf_new = TfidfVectorizer()
        X = tfidf_new.fit_transform(combined['transformed'])
        y = combined['label']

        # 5️⃣ Train
        st.write("🎓 Training new model...")
        model_new = MultinomialNB()
        model_new.fit(X, y)

        # 6️⃣ Save updated model & vectorizer
        with open("model3.pkl", "wb") as f:
            pickle.dump(model_new, f)
        with open("vectorizer3.pkl", "wb") as f:
            pickle.dump(tfidf_new, f)

            # 7️⃣ Clear feedback.csv to avoid reusing old corrections
            open("feedback.csv", "w").close()

            st.success(
                "✅ Retraining complete! Updated model3.pkl and vectorizer3.pkl saved. Feedback log cleared for new corrections.")

        st.success("✅ Retraining complete! Updated model3.pkl and vectorizer3.pkl saved.")

    except Exception as e:
        st.error(f"❌ Retraining failed: {e}")


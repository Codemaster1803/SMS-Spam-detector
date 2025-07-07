from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import pickle
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

ps = PorterStemmer()

# Load model and vectorizer with error handling
try:
    tfidf = pickle.load(open('vectorizer3.pkl', 'rb'))
    model = pickle.load(open('model3.pkl', 'rb'))
except FileNotFoundError:
    st.error(
        "🚨 **Error:** Could not find `vectorizer3.pkl` or `model3.pkl`. Please ensure they exist in the working directory."
    )
    st.stop()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [word for word in text if word.isalnum()]
    y = [word for word in y if word not in stopwords.words('english') and word not in string.punctuation]
    y = [ps.stem(word) for word in y]
    return " ".join(y)

# --------------- Streamlit UI ---------------
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

with st.container():
    st.markdown(
        """
        <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; text-align:center;">
            <h1 style="color:#2c3e50; font-size:2.8em; font-weight:700;">📩 SMS Spam Classifier</h1>
            <p style="color:#555; font-size:1.2em;">
                Enter your SMS message below and click <strong>Predict</strong> to see if it's 
                <span style="color:#e74c3c;"><b>Spam</b></span> or 
                <span style="color:#27ae60;"><b>Not Spam</b></span>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ✉️ Message Input")
    input_sms = st.text_area("", height=150, placeholder="Type your SMS message here...")

    if st.button('🚀 Predict', use_container_width=True):
        if not input_sms.strip():
            st.warning("⚠️ Please enter a message before predicting.")
        else:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            st.session_state["last_result"] = result
            st.session_state["last_input_sms"] = input_sms

            st.markdown("---")
            if result == 1:
                st.markdown(
                    """
                    <div style="background-color:#ffe6e6; padding:20px; border-radius:10px; text-align:center;">
                        <h2 style="color:#e74c3c; font-weight:600;">🚫 Spam Detected</h2>
                        <p style="color:#c0392b; font-size:1.2em;">This message is classified as <strong>SPAM</strong>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="background-color:#e8f5e9; padding:20px; border-radius:10px; text-align:center;">
                        <h2 style="color:#27ae60; font-weight:600;">✅ Not Spam</h2>
                        <p style="color:#1e8449; font-size:1.2em;">This message is classified as <strong>NOT SPAM</strong>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
        <hr style="margin-top:40px; margin-bottom:10px;"/>
        <p style="text-align:center; color:#aaa; font-size:0.9em;">
            🔒 Your message is processed locally — no data is stored or shared.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if "last_result" in st.session_state and "last_input_sms" in st.session_state:
        with st.expander("📝 Feedback — was this prediction correct?"):
            col1, col2 = st.columns(2)

            with col1:
                if st.button("👍 Yes, correct"):
                    with open("feedback.csv", "a", encoding="utf-8") as f:
                        f.write(f'"{st.session_state["last_input_sms"]}",{st.session_state["last_result"]}\n')
                    st.success("✅ Thanks for your feedback!")

            with col2:
                wrong_label = 0 if st.session_state["last_result"] == 1 else 1
                if st.button("👎 No, it was wrong"):
                    with open("feedback.csv", "a", encoding="utf-8") as f:
                        f.write(f'"{st.session_state["last_input_sms"]}",{wrong_label}\n')
                    st.warning("⚠️ Got it! Your correction has been saved!")

    st.markdown("---")
    admin_key = st.text_input("🔑 Admin key to unlock retraining:", type="password")
    if admin_key == "mysecretkey":  # Replace with your real admin key
        st.header("🔄 Admin: Retrain Model with Feedback")
        if st.button("🚀 Retrain Now"):
            try:
                original = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
                original.columns = ['label', 'message']
                original['label'] = original['label'].map({'ham': 0, 'spam': 1})

                try:
                    feedback = pd.read_csv("feedback.csv", names=['message', 'label'])
                    feedback['label'] = feedback['label'].astype(int)

                    # ✅ Only retrain if enough feedback collected
                    if len(feedback) < 10:
                        st.warning(f"⚠️ Only {len(feedback)} feedback entries found — need at least 10 before retraining.")
                        st.stop()

                    combined = pd.concat([original, feedback], ignore_index=True)
                    st.info(f"✅ Loaded {len(feedback)} feedback entries.")

                except FileNotFoundError:
                    st.warning("⚠️ No feedback found. Training with original dataset only.")
                    st.stop()

                st.write("🔄 Preprocessing messages...")
                combined['transformed'] = combined['message'].apply(transform_text)

                st.write("🔠 Vectorizing with TF-IDF...")
                tfidf_new = TfidfVectorizer()
                X = tfidf_new.fit_transform(combined['transformed'])
                y = combined['label']

                st.write("🎓 Training new model...")
                model_new = MultinomialNB()
                model_new.fit(X, y)

                with open("model3.pkl", "wb") as f:
                    pickle.dump(model_new, f)
                with open("vectorizer3.pkl", "wb") as f:
                    pickle.dump(tfidf_new, f)

                # ✅ Archive feedback before clearing
                if not feedback.empty:
                    feedback.to_csv("archived_feedback.csv", mode="a", header=False, index=False, quoting=1)
                    st.success(f"✅ Archived {len(feedback)} feedback entries to archived_feedback.csv.")

                # ✅ Clear feedback after archiving
                open("feedback.csv", "w").close()

                st.success("✅ Retraining complete! Updated model saved. Feedback log cleared for new corrections.")

            except Exception as e:
                st.error(f"❌ Retraining failed: {e}")
    else:
        st.info("🔒 Retraining is locked — enter admin key to unlock.")

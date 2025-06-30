import streamlit as st
import pickle

import nltk


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
# Load model and vectorizer with error handling
try:
    tfidf = pickle.load(open('vectorizer3.pkl', 'rb'))
    model = pickle.load(open('model3.pkl', 'rb'))
except FileNotFoundError:
    st.error(
        "🚨 **Error:** Could not find `vectorizer3.pkl` or `model3.pkl`. Please ensure they exist in the working directory."
    )
    st.stop()

# Preprocessing imports
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


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
        <h1 style="color:#4a90e2; text-align: center; font-size: 2.5em; font-weight:600;">📩 SMS Spam Classifier</h1>
        <p style="text-align: center; color: #666; font-size: 1.1em;">
            Enter your SMS message below and click <strong>Predict</strong> to see if it's <span style="color:#e74c3c;"><b>Spam</b></span> or <span style="color:#27ae60;"><b>Not Spam</b></span>.
        </p>
        """,
        unsafe_allow_html=True,
    )

    input_sms = st.text_area("✉️ **Message Text**", height=150, placeholder="Type your SMS message here...")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button('Predict', use_container_width=True):
            if not input_sms.strip():
                st.warning("⚠️ Please enter a message before predicting.")
            else:
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]

                st.markdown("---")
                if result == 1:
                    st.markdown(
                        """
                        <div style="background-color: #ffe6e6; padding: 20px; border-radius: 10px; text-align: center;">
                            <h2 style="color: #e74c3c;">🚫 Spam Detected</h2>
                            <p style="color: #b03a2e; font-size: 1.2em;">This message is classified as <strong>SPAM</strong>.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; text-align: center;">
                            <h2 style="color: #27ae60;">✅ Not Spam</h2>
                            <p style="color: #1e8449; font-size: 1.2em;">This message is classified as <strong>NOT SPAM</strong>.</p>
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

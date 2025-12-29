import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords

# -------------------------------
# Load stopwords (cached)
# -------------------------------
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# -------------------------------
# Load model & vectorizer (cached)
# -------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer

# -------------------------------
# Sentiment prediction function
# -------------------------------
def predict_sentiment(text, model, vectorizer, stop_words):
    # Clean text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)

    # Transform and predict
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]

    return "Negative" if prediction == 0 else "Positive"

# -------------------------------
# Main Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
    st.title("🧠 Sentiment Analysis App")
    st.write("Analyze sentiment of any text using a trained ML model.")

    # Load resources
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    # Text input
    user_text = st.text_area("Enter text to analyze sentiment")

    if st.button("Analyze Sentiment"):
        if user_text.strip():
            sentiment = predict_sentiment(user_text, model, vectorizer, stop_words)

            if sentiment == "Positive":
                st.success("😊 Positive Sentiment")
            else:
                st.error("😠 Negative Sentiment")
        else:
            st.warning("Please enter some text.")

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    main()

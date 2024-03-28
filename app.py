import streamlit as st
import joblib

# Load the model and vectorizer
classifier, vectorizer = joblib.load('spam_classifier_model.joblib')

# Define function to predict message
def predict_message(message):
    message_vectorized = vectorizer.transform([message])
    prediction = classifier.predict(message_vectorized)
    return prediction[0]

# Set page title and favicon
st.set_page_config(
    page_title="Spam Detection App",
    page_icon=":email:"
)

# Header
st.title("Spam Detection App")
st.write("Enter an SMS below and check if it's spam.")

# SMS input
sms = st.text_area("Enter an SMS")

if sms:
    prediction = predict_message(sms)

    # Spam detection results
    st.write("## Spam Detection Results")

    if prediction == 1:
        st.error("This SMS might be spam.")
    else:
        st.success("This SMS is not spam.")

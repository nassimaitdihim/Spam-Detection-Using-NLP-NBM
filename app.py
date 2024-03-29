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

import streamlit as st

# Sidebar content
sidebar_content = """
# Ecole Supérieure de Technologie Fkih Ben Salah

---

**Équipe de réalisation :**

- 👨‍🎓 &nbsp;&nbsp;&nbsp;&nbsp; Nassim Aït Dihim
- 👨‍🎓 &nbsp;&nbsp;&nbsp;&nbsp; Said Tallouk
- 👨‍🎓 &nbsp;&nbsp;&nbsp;&nbsp; Anass Nabil
- 👨‍🎓 &nbsp;&nbsp;&nbsp;&nbsp; Mohammed Laalahmi

---

**Demandé Par :**

- 👨‍🏫 &nbsp;&nbsp;&nbsp;&nbsp; Pr. Hassan Fouazi
"""

# Load the image
image = 'estfbs logo.png'  # Replace with the path to your image

# Display the image in the sidebar
st.sidebar.image(image, width=150)

# Display sidebar content
st.sidebar.markdown(sidebar_content)

# Header
st.title("📧 Spam Detection App 💥")
st.info("Enter an SMS below and check if it's spam.")

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

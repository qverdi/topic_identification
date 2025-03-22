import streamlit as st
import re
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Class labels
class_labels = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load model and tokenizer once
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# App layout
st.title("🧠 Research Article Category Classifier")
st.markdown("Enter the title and abstract below to predict the research field.")

# Input fields
title_input = st.text_input("Title")
abstract_input = st.text_area("Abstract")

# Submit button
if st.button("Submit"):
    if not title_input or not abstract_input:
        st.warning("Please fill in both the title and abstract.")
    else:
        # Clean and combine input
        full_text = clean_text(title_input + " " + abstract_input)
        
        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([full_text])
        padded = pad_sequences(seq, maxlen=300)

        # Predict
        prediction = model.predict(padded)
        predicted_class = class_labels[np.argmax(prediction)]

        # Show result
        st.success(f"🔍 Predicted Category: **{predicted_class}**")

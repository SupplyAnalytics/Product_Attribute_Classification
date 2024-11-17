# app.py
import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from image_vectors import extract_feature_vector

# # Load the pre-trained model
# MODEL_PATH = "sgd_model_unscaled.pkl"
# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "sgd_model_unscaled.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Subcategory Prediction App :shorts::shirt::jeans::kimono::bikini::tshirt::dress::shoe::boot::high_heel::sandal:")

st.title(":yellow_heart: Hello Bijnicians... :sunglasses:")
st.write("Hope you're doing well and enjoying your work today...:100:")
st.write("Upload an image, and the app will predict the output using the pre-trained model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=False)

    # Save the uploaded image temporarily
    temp_image_path = "temp_uploaded_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Processing the image...")

    # Extract feature vector
    try:
        vector = extract_feature_vector(temp_image_path)
        st.write("Image feature vector extracted successfully.")

        # Predict using the loaded model
        prediction = model.predict([vector])
        st.write(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Cleanup temporary file
    os.remove(temp_image_path)

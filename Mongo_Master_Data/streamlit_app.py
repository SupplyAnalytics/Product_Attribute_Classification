# # app.py
# import streamlit as st
# import os
# import pickle
# import numpy as np
# from PIL import Image
# from image_vectors import extract_feature_vector

# # # Load the pre-trained model
# # MODEL_PATH = "sgd_model_unscaled.pkl"
# # with open(MODEL_PATH, "rb") as f:
# #     model = pickle.load(f)

# MODEL_PATH = os.path.join(os.path.dirname(__file__), "sgd_model_unscaled.pkl")
# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)

# # Streamlit UI
# st.title("SPA: Subcategory Prediction App")
# st.title(":shorts::shirt::jeans::kimono::shoe::boot::bikini::tshirt::dress::high_heel::sandal:")
# st.title(":yellow_heart: Hello Bijnicians... :sunglasses:")
# st.write("Upload an image, and the app will predict the subcategory of the article...*Voilla*")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image from gallery...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

#     # Save the uploaded image temporarily
#     temp_image_path = "temp_uploaded_image.jpg"
#     with open(temp_image_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     st.write("Processing the image...")

#     # Extract feature vector
#     try:
#         vector = extract_feature_vector(temp_image_path)
#         st.write("Image feature vector extracted successfully.")

#         # Predict using the loaded model
#         prediction = model.predict([vector])
#         st.title(f"Prediction: {prediction[0]}")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")

#     # Cleanup temporary file
#     os.remove(temp_image_path)


# app.py
import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from image_vectors import extract_feature_vector

# Load the pre-trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sgd_model_unscaled.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("SPA: Subcategory Prediction App")
st.title(":shorts::shirt::jeans::kimono::shoe::boot::bikini::tshirt::dress::high_heel::sandal:")
st.title(":yellow_heart: Hello Bijnicians... :sunglasses:")
st.write("Upload an image, or provide the path of a folder containing multiple images, and the app will predict the subcategory of the articles...*Voilla*")

# Option 1: Upload an image
uploaded_file = st.file_uploader("Choose an image from gallery...", type=["jpg", "jpeg", "png"])

# Option 2: Provide folder path for multiple images
folder_path = st.text_input("Or, provide the folder path containing multiple images")

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save the uploaded image temporarily
    temp_image_path = "temp_uploaded_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Processing the image...")

    # Extract feature vector and make prediction
    try:
        vector = extract_feature_vector(temp_image_path)
        st.write("Image feature vector extracted successfully.")

        # Predict using the loaded model
        prediction = model.predict([vector])
        st.title(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Cleanup temporary file
    os.remove(temp_image_path)

elif folder_path:
    # Validate the folder path and process each image
    if os.path.isdir(folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)

                # Display each image
                st.image(image_path, caption=f"Image: {image_file}", use_container_width=True)

                st.write(f"Processing image: {image_file}...")

                try:
                    # Extract feature vector
                    vector = extract_feature_vector(image_path)
                    st.write(f"Image feature vector extracted successfully for {image_file}.")

                    # Predict using the loaded model
                    prediction = model.predict([vector])
                    st.title(f"Prediction for {image_file}: {prediction[0]}")
                except Exception as e:
                    st.error(f"An error occurred while processing {image_file}: {e}")
        else:
            st.error("No valid image files found in the specified folder.")
    else:
        st.error("Invalid folder path.")

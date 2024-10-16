class Image_Vectorization:

    import os
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.preprocessing.image import img_to_array
    from PIL import Image, UnidentifiedImageError
    import requests
    from io import BytesIO
    import streamlit as st
    import json
    import datetime


    def extract_feature_vector(image_path):
        # Load the pre-trained MobileNetV2 model
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

        # Open the image using PIL
        img = Image.open(image_path)

        if img.format != 'JPEG':
            # Convert image to RGB if it is not
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.save(image_path, format='JPEG')

        # Resize the image to the size expected by MobileNetV2 (224x224)
        img = img.resize((224, 224))
        # Convert the image to a numpy array and preprocess for the model
        img_array = np.array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Extract the feature vector using the model
        feature_vector = model.predict(img_array)

        return feature_vector

    # Example usage
    image_path = r"C:\Users\Admin\Attribute_Classification\Product_Attribute_Classification\Image_Vectorization\Image_1.jpeg"
    feature_vector = extract_feature_vector(image_path)
    image_vector_df = pd.DataFrame(feature_vector)
    image_vector_df.columns = [f"V{i+1}" for i in range (image_vector_df.shape[1])]
    image_vector_df



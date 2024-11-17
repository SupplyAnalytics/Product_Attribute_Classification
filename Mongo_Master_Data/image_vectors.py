

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import json
import datetime


# def extract_feature_vector(image_path):
#     # Load the pre-trained MobileNetV2 model
#     base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
#     model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

#     # Open the image using PIL
#     img = Image.open(image_path)

#     if img.format != 'JPEG':
#         # Convert image to RGB if it is not
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
        
#         img.save(image_path, format='JPEG')

#     # Resize the image to the size expected by MobileNetV2 (224x224)
#     img = img.resize((224, 224))
#     # Convert the image to a numpy array and preprocess for the model
#     img_array = np.array(img)
#     img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Extract the feature vector using the model
#     feature_vector = model.predict(img_array)

#     return feature_vector

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

    # Flatten the feature vector to 1D
    return feature_vector.flatten()


# # Example usage
# image_path = r"C:\Users\Admin\Attribute_Classification\Product_Attribute_Classification\Image_Vectorization\Image_1.jpeg"
# feature_vector = extract_feature_vector(image_path)
# feature_vector



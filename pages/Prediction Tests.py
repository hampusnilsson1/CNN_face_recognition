import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
#View Test Predictions

st.header("Prediction Test From Images.")
prediction_placeholder = st.empty()

def predictAndDraw(img_path, size):
    img = Image.open(img_path).convert('L') 
    img = img.resize((48, 48))
    img_array = image.img_to_array(img)  # Convert the image to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch (1, 224, 224, 3)
    print()
    img_array = img_array / 255.0  # Normalize the image (if needed)
    
    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions, axis=1)
    
    class_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise",}
    predicted_label = class_labels[predicted_class[0]]
    
    return img, predicted_label

def display_images_in_grid(images_with_labels, size, columns=3):
    cols = st.columns(columns)
    
    for index, (img, label) in enumerate(images_with_labels):
        with cols[index % columns]:
            st.image(img, caption=f"Predicted: {label}", width=size)

model = load_model("models/facial_emotion_model.h5")
test_pictures_path = "ExtraTest/"
images_with_labels = []

for filename in os.listdir(test_pictures_path):
    file_path = os.path.join(test_pictures_path, filename)
    
    # Ensure that it's a file and not a directory
    if os.path.isfile(file_path):
        img, predicted_label = predictAndDraw(file_path, 200)
        images_with_labels.append((img, predicted_label))
        
display_images_in_grid(images_with_labels, 180, columns=4)
st.subheader("Model Training Visualisation")
st.image("train_val_acc_loss.png")
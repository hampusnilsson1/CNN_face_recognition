import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

def view_random_image(target_dir, target_class, frame, size):
    target_folder = target_dir +"/"+ target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    img = mpimg.imread(target_folder + '/' + random_image[0])
    frame.image(img, width=size)

st.header("Data View")
    
expression = st.select_slider(
    "Select the expression training you want to view.",
    options=[
        "Happy",
        "Surprise",
        "Neutral",
        "Disgust",
        "Fear",
        "Sad",
        "Angry",
    ],
)
st.write("Showing: ", expression)

picture_placeholder = st.empty()

view_random_image("data/train", expression, picture_placeholder,300)
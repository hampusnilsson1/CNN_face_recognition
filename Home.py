import streamlit as st
from streamlit_webrtc import webrtc_streamer

import numpy as np
import tensorflow
import cv2
import av

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.header("Facial Details Recognition Webcam")

checks = st.columns(3)
with checks[0]:
    expression_check = st.checkbox("Detect Expression")
with checks[1]:
    gender_check = st.checkbox("Detect Gender")
with checks[2]:
    ethnicity_check = st.checkbox("Detect Ethnicity")
st.markdown("(Reload to apply changes)")

class VideoProcessor:
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.expression_model = load_model("models/facial_emotion_model.h5")
        self.gender_model = load_model("models/gender_model.h5")
        self.ethnicity_model = load_model("models/ethnicity_model.h5")

        self.expression_labels = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
        self.gender_labels = ["Male", "Female"]
        self.ethnicity_labels = ["White", "Black", "Asian", "Indian", "Others"]

        self.expression_check = expression_check
        self.gender_check = gender_check
        self.ethnicity_check = ethnicity_check
        pass

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            print("Got image")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print("Turned gray")
            faces = self.face_classifier.detectMultiScale(gray)
            print("Detected face loaded")

            for (x, y, w, h) in faces:
                if w > 100:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (153, 255, 153), 1)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray]) != 0:
                        roi = np.expand_dims(roi_gray, axis=0)
                        roi = img_to_array(roi)
                        roi_expression = roi.astype("float") / 255.0

                        if expression_check:
                            expression_prediction = self.expression_model.predict(roi_expression)[0]
                            expression_label = self.expression_labels[expression_prediction.argmax()]
                            expression_label_position = (x, y - 10)

                            label_color = self.get_expression_label_color(expression_label)
                            cv2.putText(img, expression_label, expression_label_position, cv2.FONT_HERSHEY_DUPLEX, 1, label_color, 2)

                        if gender_check:
                            gender_prediction = self.gender_model.predict(roi)[0]
                            odds = int(100 - gender_prediction[0] * 100) if gender_prediction < 0.5 else int(gender_prediction[0] * 100)
                            gender_prediction = 0 if gender_prediction < 0.5 else 1
                            gender_label_color = (204, 204, 0) if gender_prediction == 0 else (204, 0, 204)

                            gender_label = self.gender_labels[gender_prediction]
                            gender_label_position = (x, y - 40) if expression_check else (x, y - 10)
                            cv2.putText(img, f"{gender_label}({odds}%)", gender_label_position, cv2.FONT_HERSHEY_DUPLEX, 1, gender_label_color, 2)

                        if ethnicity_check:
                            ethnicity_prediction = self.ethnicity_model.predict(roi)[0]
                            ethnicity_label = self.ethnicity_labels[ethnicity_prediction.argmax()]
                            ethnicity_label_position = (x, y + h + 30)
                            cv2.putText(img, ethnicity_label, ethnicity_label_position, cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return av.VideoFrame.from_ndarray(img, format="rgb24")
        
        except Exception as e:
            print(f"Error converting frame to ndarray: {e}")
            return frame

    def get_expression_label_color(self, expression_label):
        colors = {
            "Angry": (0, 0, 153),
            "Disgust": (40, 40, 63),
            "Fear": (102, 51, 0),
            "Happy": (13, 190, 42),
            "Neutral": (204, 255, 255),
            "Sad": (92, 64, 46),
            "Surprise": (1, 191, 250)
        }
        return colors.get(expression_label, (255, 255, 255))

webrtc_streamer(key="edge", video_processor_factory=VideoProcessor)
#cd C:\Users\Hampus\Documents\Python Scripts\kunskapskontroll_2_2_hampus_nilsson
#streamlit run Home.py

import streamlit as st

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import numpy as np
import tensorflow
import cv2

from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

#Streamlit
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
expression_model = load_model("models/facial_emotion_model.h5")
gender_model = load_model("models/gender_model.h5")
ethnicity_model = load_model("models/ethnicity_model.h5")

expression_labels = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
gender_labels = ["Male", "Female"]
age_labels = ["1-14", "14-28", "29-43", "44-58", "59-73", "74-87", "88-100", "100+"]
ethnicity_labels = ["White", "Black", "Asian", "Indian", "Others"]

st.header("Facial Details Recognition Webcam")

#Create Camera Capturing
allow_camera = st.checkbox("Activate Camera")
frame_placeholder = st.empty()

checks = st.columns(3)
with checks[0]:
    expression_check = st.checkbox("Detect Expression")
with checks[1]:
    gender_check = st.checkbox("Detect Gender")
with checks[2]:
    ethnicity_check = st.checkbox("Detect Ethnicity")


if allow_camera:
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # Break if no capture
        if not ret:
            st.error("Error: No image was found!")
            break
    
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
    
        for(x,y,w,h) in faces:
            if w > 100:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(153, 255, 153),1) #Color of Rectangle aswell as size depending on detection
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
               
                if np.sum([roi_gray]) != 0:
                    roi = np.expand_dims(roi_gray,axis=0)
                    roi = img_to_array(roi)
                    roi_expression = roi.astype("float")/255.0
                    
                    #Expression Predict
                    if expression_check:
                        expression_predition = expression_model.predict(roi_expression)[0]
                        
                        expression_label=expression_labels[expression_predition.argmax()]
                        expression_label_position = (x,y-10)
        
                        match expression_label:
                            case "Angry":     #BLUE GREEN RED
                                label_color = (0, 0, 153)
                            case "Disgust":
                                label_color = (40, 40, 63)
                            case "Fear":
                                label_color = (102, 51, 0)
                            case "Happy":
                                label_color = (13, 190, 42)
                            case "Neutral":
                                label_color = (204, 255, 255)
                            case "Sad":
                                label_color = (92, 64, 46)
                            case "Surprise":
                                label_color = (1, 191, 250)
                                
                        cv2.putText(frame,expression_label,expression_label_position,cv2.FONT_HERSHEY_DUPLEX,1,label_color,2)
                        
                    #Gender Predict 
                    if gender_check:
                        gender_prediction = gender_model.predict(roi)[0]  # Get the first prediction
                        if gender_prediction < 0.5:
                            odds = 100 - gender_prediction[0]*100
                            odds = odds.astype(int)
                            gender_prediction = 0
                            gender_label_color = (204,204,0)
                        else:
                            odds = gender_prediction[0]*100
                            odds = odds.astype(int)
                            gender_prediction = 1
                            gender_label_color = (204,0,204)
                            
                        gender_label = gender_labels[gender_prediction]
                        if expression_check:
                            gender_label_position = (x,y-40)
                        else:
                            gender_label_position = (x,y-10)
                        
                        # Display the label on the frame
                        cv2.putText(frame, f"{gender_label}({odds}%)",
                                    gender_label_position, cv2.FONT_HERSHEY_DUPLEX, 1, gender_label_color, 2)
                       
                    #Ethnicity Predict
                    if ethnicity_check:      
                        ethnicity_prediction = ethnicity_model.predict(roi)[0]  # Get the first prediction
                        
                        ethnicity_label = ethnicity_labels[ethnicity_prediction.argmax()]
                        ethnicity_label_position = (x,y+h+30)#(x,y+h+30)#(x,y-10)
                        
                        # Display the label on the frame
                        cv2.putText(frame, ethnicity_label, ethnicity_label_position, cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    
        #Turn to color before showing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the frame in the Streamlit app
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    cap.release()
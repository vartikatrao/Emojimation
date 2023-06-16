import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image

st.title('Facial emotion recognition')

# loading our deep learning model
emotion_model = keras.models.load_model('FER2013new.h5')

# emotion relation with int
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# classifier for face identification
bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# get image by uploading a file 
img_uploader = st.file_uploader("upload an image",type = ['jpg','png','jpeg'])

if img_uploader is not None:
    st.image(img_uploader)
    img = np.array(Image.open(img_uploader))
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
    if num_faces == ():
        st.write("face in the image should be straight")
    
    for x, y, w, h in num_faces:
        cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        st.write(emotion_dict[maxindex])

# get image by directly capturing from cam
img_cap = st.camera_input("switch on camera")

if img_cap is not None:
    img = np.array(Image.open(img_cap))
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
    if num_faces == ():
        st.write("keep your face straight")
        
    for x, y, w, h in num_faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        st.write(emotion_dict[maxindex])
        

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer,WebRtcMode
import cv2
import os
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
age = os.path.join(ROOT_DIR,'data/models',"AgePrediction.h5")
gender = os.path.join(ROOT_DIR,'data/models',"GenderPrediction1.h5")
classifier = os.path.join(ROOT_DIR,'data',"haarcascade_frontalface_alt.xml")

import streamlit as st
st.title("Age Gender Prediction")

gender_model = keras.models.load_model(gender)

def predict_image(predict_img):
    x = cv2.resize(predict_img, (48, 48))
    x = x.reshape(-1, 48, 48, 1)
    test_image = ImageDataGenerator(rescale=1. / 255.0).flow(x)
    pred_value = np.argmax(gender_model.predict(test_image))
    if pred_value == 0:
        return "Male"
    if pred_value == 1:
        return "Female"

age_model = keras.models.load_model(age)
def age_image(predict_img):
    x = cv2.resize(predict_img, (48, 48))
    x = x.reshape(-1,48,48,1)
    op = ImageDataGenerator(rescale=1./255.0).flow(x)
    predictions = age_model.predict(op)
    return predictions


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        face_cascade = cv2.CascadeClassifier(classifier)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            gray_face_img = gray[y:y + h, x:x + w].copy()

            # Gender Prediction
            gender = predict_image(gray_face_img)
            cv2.putText(image, gender, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Age Prediction
            age = age_image(gray_face_img)
            #min_value = round(age[0][0]) - 3
            #max_value = round(age[0][0]) + 3
            cv2.putText(image, str(round(age[0][0])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return image

webrtc_streamer(key="example", video_processor_factory=VideoTransformer,rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False})
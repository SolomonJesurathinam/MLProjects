import cv2
import os
import streamlit as st
import numpy as np
import requests


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
classifier = os.path.join(ROOT_DIR,'data',"haarcascade_frontalface_alt.xml")

def radio_functiom(input_image):
    if input_image is not None:
        bytes_data = input_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        face_cascade = cv2.CascadeClassifier(classifier)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            color_face_img = cv2_img[y:y + h, x:x + w].copy()

            success, encoded_image = cv2.imencode('.png', color_face_img)  #important when transferring images
            files = {"file": encoded_image.tobytes()}
            res = requests.post("https://classifier-api.herokuapp.com/predict/gender", files=files)
            output = res.json()
            cv2.putText(cv2_img, str(output["Predicted Gender"]), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        st.image(cv2_img)


radio_values = st.radio(label="Prediction",options= ("Upload a picture","Photo from Camera"))
if radio_values == "Upload a picture":
    input_image = st.file_uploader("Upload a pic")
    output_image = radio_functiom(input_image)

if radio_values == "Photo from Camera":
    input_image = st.camera_input("Take a Picture to predict")
    radio_functiom(input_image)








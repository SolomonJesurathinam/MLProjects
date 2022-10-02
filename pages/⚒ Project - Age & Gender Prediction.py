import cv2
import os
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
age = os.path.join(ROOT_DIR,'data/models',"age_net.caffemodel")
age_proto = os.path.join(ROOT_DIR,'data/models',"deploy_age.prototxt")
gender = os.path.join(ROOT_DIR,'data/models',"GenderPrediction1.h5")
classifier = os.path.join(ROOT_DIR,'data',"haarcascade_frontalface_alt.xml")

import streamlit as st
st.title("Age Gender Prediction")

gender_model = keras.models.load_model(gender,compile=False)
age_model = cv2.dnn.readNet(age_proto, age)
AGE_RANGE = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)","(38-43)", "(48-53)", "(60-100)"]

def predict_image(predict_img):
    x = cv2.resize(predict_img, (48, 48))
    x = x.reshape(-1, 48, 48, 1)
    test_image = ImageDataGenerator(rescale=1. / 255.0).flow(x)
    pred_value = np.argmax(gender_model.predict(test_image))
    if pred_value == 0:
        return "Male"
    if pred_value == 1:
        return "Female"

def age_model_func(image):
    #face = cv2.imread(image)
    faceBlob = cv2.dnn.blobFromImage(image, 1.0, (227, 227),(78.4263377603, 87.7689143744, 114.895847746),swapRB=False)
    age_model.setInput(faceBlob)
    preds = age_model.forward()
    i = preds[0].argmax()
    age = AGE_RANGE[i]
    ageConfidence = preds[0][i]
    text = "{}: {:.2f}%".format(age, ageConfidence * 100)
    return text


def predict(image):
    face_cascade = cv2.CascadeClassifier(classifier)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        gray_face_img = gray[y:y + h, x:x + w].copy()
        color_face_img = image[y:y + h, x:x + w].copy()

        # Gender Prediction
        gender = predict_image(gray_face_img)
        cv2.putText(image, gender, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Age Prediction
        age = age_model_func(color_face_img)
        cv2.putText(image, age, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    return image

def radio_functiom(input_image):
    if input_image is not None:
        bytes_data = input_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        output = predict(cv2_img)
        st.image(output)

radio_values = st.radio(label="Prediction",options= ("Upload a picture","Photo from Camera"))
if radio_values == "Upload a picture":
    input_image = st.file_uploader("Upload a pic",type=['png', 'jpg'])
    radio_functiom(input_image)

if radio_values == "Photo from Camera":
    input_image = st.camera_input("Take a Picture to predict")
    radio_functiom(input_image)


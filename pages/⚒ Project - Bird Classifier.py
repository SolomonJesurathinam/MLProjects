import streamlit as st
import requests

image = st.file_uploader("Choose an image")
if image is not None:
    files = {"file":image.getvalue()}
    res = requests.post("https://birdclassifier-api.herokuapp.com/predict",files=files)
    output = res.json()
    st.image(image)
    st.success(output["Predicted Bird"])
    st.write("Confidence", output["Confidence"])
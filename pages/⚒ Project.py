import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
#st.sidebar.write("Details about Project")


#read excel
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
excel_path = os.path.join(ROOT_DIR,'data',"Crop_recommendation.csv")
data = pd.read_csv(excel_path)
#print(data.head())

X = data.iloc[:, 0:7]
Y = data["label"]

def selectModel():
    # split data

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    # Gaussian Model
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    return accuracy

st.set_page_config(initial_sidebar_state="auto",page_title="Crop Recommendation",page_icon="🌱")
st.header("Crop Recommendation🌱")

st.subheader("Find out the suitable crop for your farm 👨‍🌾")

N = st.number_input("Nitrogen",min_value=1,max_value=10000,step=1)
P = st.number_input("Phosporus",min_value=1,max_value=10000,step=1)
K = st.number_input("Potassium",min_value=1,max_value=10000,step=1)
temperature = st.number_input("Temperature",min_value=0.00,max_value=100000.00)
humidity = st.number_input("Humidity in %",min_value=0.00,max_value=100000.00)
ph = st.number_input("Ph",min_value=0.00,max_value=100000.00)
rainfall = st.number_input("Rainfall in mm",min_value=0.00,max_value=100000.00)

xvalue = [[N,P,K,temperature,humidity,ph,rainfall]]
model = GaussianNB()
model.fit(X,Y)

if st.button("Predict"):
    y_predict = model.predict(xvalue)
    st.success("{} are recommended by A.I".format(y_predict[0].capitalize()))
st.warning("Note: This A.I application is just for educational/demo purposes only and cannot be relied upon.")


with st.sidebar.expander("ℹ Information", expanded=True):
    st.write('''Crop recommendation is one of the most important aspects of precision agriculture. 
                    Most output from agriculture comes from small farms that routinely choose crops that have been sown in the same land for generations. 
                    But their crop selections are not always the most optimal. In this project we build a crop recommender system to solve such problems.
                    We are using Gaussian NB model for the prediction.The final aim is to suggest the most suitable crop to farmers based on minerals and climate conditions.''')

st.sidebar.subheader("How this will work ❓")
st.sidebar.write("Complete all the parameters and the ML model will predict the most suitable crops to grow based on various parameters")

with st.sidebar.expander("Model",expanded=False):
    if st.button("Get the Model & Accuracy"):
        accuracy = selectModel()
        st.write("Model = GaussianNB")
        st.write("Accuracy = {}%".format(round(accuracy*100,2)))
        st.warning("Note: Model and accuracy score is used for demo/educational purposes")


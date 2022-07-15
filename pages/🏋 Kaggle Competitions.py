import random

from pages.Kaggle import Titanic_Survivors as ts
import os
import streamlit as st

def title():
    st.set_page_config(page_title="Kaggle Competitions", layout='centered', initial_sidebar_state='auto',
                       page_icon='💣')
    st.header("🏋 Kaggle Competitions 🏋")

def dropdown():
    # Dropdown options
    global options
    options = st.selectbox(label="Select Project", options=[" ", "Titanic", "Space"], index=0)
    if options == " ":
        st.write("Please select a project from dropdown")

#Titanic code
def titanic():
    if options == "Titanic":
        #sidebar functions
        st.sidebar.info("Using machine learning to create a model that predicts which passengers survived the Titanic shipwreck.")
        with st.sidebar.expander("Coding Links",expanded=False):
            st.write("[Link](https://www.kaggle.com/code/solomonyolo/titanic-wip) to Kaggle Notebook code")
            st.write("[Link](https://github.com/SolomonJesurathinam/JuypterProjects/tree/master/Kaggle%20Competitions/Titanic) to full code from Github ")
        #image function
        ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
        image_path = os.path.join(ROOT_DIR, 'data', "KaggleTitanic.png")
        with st.sidebar.expander("Kaggle Public score",expanded=False):
            from PIL import Image
            kaggleImage = Image.open(image_path)
            st.image(kaggleImage)
        #Caching the model values
        @st.cache
        def cacheFunc(a):
            test = ts.Titanic()  # object for Titanic class
            train_data, test_data = test.load_data()
            processed_data = test.data_processing()
            forest_accuracy, decision_accuracy,xgb_accuracy = test.model_accuracy()
            output = test.realtime_data()
            return train_data,test_data,processed_data,forest_accuracy, decision_accuracy,xgb_accuracy,output

        #model reset
        cacheValue=1
        cache = st.button("Reset Cache",help="Reset's the model cached value and build again - Time Consuming")
        if cache:
            cacheValue = cacheValue + random.randint(0, 9)

        #Model UI elements
        train_data,test_data,processed_data,forest_accuracy,decision_accuracy,xgb_accuracy,output = cacheFunc(cacheValue)
        load_data = st.checkbox("Training Data")
        if load_data:
            st.write(train_data)
        processedData = st.checkbox("Processed Training Data")
        if processedData:
            st.write(processed_data)
        modelAccuracy = st.checkbox("Model Accuracy")
        if modelAccuracy:
            st.write("Forest accuracy score is: ",forest_accuracy)
            st.write("Decision Tree accuracy score is: ",decision_accuracy)
            st.write("XGBoost accuracy score is: ",xgb_accuracy)
        realData = st.checkbox("Real time data")
        if realData:
            st.write(test_data)
        outputDF=st.checkbox("Predicted output")
        if outputDF:
            st.write(output)

title()
dropdown()
titanic()
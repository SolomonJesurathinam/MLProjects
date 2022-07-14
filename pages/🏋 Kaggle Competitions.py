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
        st.sidebar.info("Using machine learning to create a model that predicts which passengers survived the Titanic shipwreck.")
        with st.sidebar.expander("Coding Links",expanded=False):
            st.write("[Link](https://www.kaggle.com/code/solomonyolo/titanic-wip) to Kaggle Notebook code")
            st.write("[Link](https://github.com/SolomonJesurathinam/JuypterProjects/tree/master/Kaggle%20Competitions/Titanic) to full code from Github ")

        ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
        image_path = os.path.join(ROOT_DIR, 'data', "KaggleTitanic.png")

        with st.sidebar.expander("Kaggle Public score",expanded=False):
            from PIL import Image
            kaggleImage = Image.open(image_path)
            st.image(kaggleImage)
        test = ts.Titanic()  # object for Titanic class
        # Raw Training Data
        train_data = st.checkbox("Data For Training")
        trainData,testData = test.load_data()
        if train_data:
            st.dataframe(trainData)

        #Training Data after Processing
        data_processed = st.checkbox("PreProcessed Data - Data Cleaning and PreProcessing", disabled=(not train_data))
        try:
            if data_processed:
                st.dataframe(test.data_processing())
        except:
            st.error("Select Raw training data and uncheck from Bottom")

        #Model Accuracy
        model_accuracy = st.checkbox("Determine Model Accuracy", disabled=(not data_processed))
        try:
            if model_accuracy:
                forest_accuracy, xgb_accuracy, decision_accuracy  = test.model_accuracy()
                st.write("Random Forest accuracy score: ", forest_accuracy)
                st.write("XGB Boost accuracy score: ", xgb_accuracy)
                st.write("Decision Tree Classifier accuracy score: ", decision_accuracy)
        except:
            st.error("Select Training Data after Processing and uncheck from Bottom")

        #Test Data
        testing_data = st.checkbox("Real time testing data",disabled=(not model_accuracy))
        if testing_data:
                st.write(testData)

        #Real time data
        output = st.checkbox("Testing the model with Real time data", disabled=(not testing_data))
        try:
            if output:
                st.dataframe(test.realtime_data())
        except:
            st.error("Select all checkboxes and uncheck from bottom")

title()
dropdown()
titanic()



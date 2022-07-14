from pages.Kaggle import Titanic_Survivors as ts
import numpy as np
import streamlit as st

def title():
    st.set_page_config(page_title="Kaggle Competitions", layout='centered', initial_sidebar_state='auto',
                       page_icon='🏋')
    st.header("🏋 Kaggle Competitions 🏋")

def dropdown():
    # Dropdown options
    global options
    options = st.selectbox(label="Select Project", options=[" ", "Titanic", "Space"], index=0)

#Titanic code
def titanic():
    if options == "Titanic":
        st.sidebar.info("Using machine learning to create a model that predicts which passengers survived the Titanic shipwreck.")
        test = ts.Titanic()  # object for Titanic class
        train_data = st.checkbox("Raw training data")
        if train_data:
            st.dataframe(test.load_data())
        data_processed = st.checkbox("Training data after Processing", disabled=(not train_data))
        if data_processed:
            st.dataframe(test.data_processing())
        model_accuracy = st.checkbox("Model Accuracy using Kfold", disabled=(not data_processed))
        if model_accuracy:
            score1, score2, score3, score4 = test.model_accuracy()
            st.write("\nRandom Forest Average score with 10 Estimators: ", np.average(score1))
            st.write("Random Forest Average score with 20 Estimators: ", np.average(score2))
            st.write("Random Forest Average score with 30 Estimators: ", np.average(score3))
            st.write("Random Forest Average score with 40 Estimators: ", np.average(score4))
        output = st.checkbox("Test on real time data", disabled=(not model_accuracy))
        if output:
            st.dataframe(test.realtime_data())


title()
dropdown()
titanic()



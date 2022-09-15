import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os

#read excel
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
excel_path = os.path.join(ROOT_DIR,'data',"FifaRankings.csv")
rankings = pd.read_csv(excel_path)

st.set_page_config(initial_sidebar_state="auto",page_title="FIFA 22 WORLD CUP PREDICTIONS",page_icon="🌱")
st.header("FIFA 22 WORLD CUP PREDICTIONS")

list_2022 = np.array(['Qatar', 'Germany', 'Denmark', 'Brazil', 'France', 'Belgium', 'Croatia', 'Spain', 'Serbia', 'England', 'Switzerland', 'Netherlands', 'Argentina', 'IR Iran', 'Korea Republic', 'Japan', 'Saudi Arabia', 'Ecuador', 'Uruguay', 'Canada', 'Ghana', 'Senegal', 'Portugal', 'Poland', 'Tunisia', 'Morocco', 'Cameroon', 'USA', 'Mexico', 'Wales', 'Australia', 'Costa Rica'])
Team1 = st.selectbox(label="Team1",options=list_2022)
Team2 = st.selectbox(label="Team2",options=list_2022)

Team1Rank = rankings[rankings["Team"] == Team1]['Rank'].to_list()[0]
Team2Rank = rankings[rankings["Team"] == Team2]['Rank'].to_list()[0]
Team1_FIFA_RANK = st.text_input(label="Team1_FIFA_RANK",value=Team1Rank,disabled=True)
Team2_FIFA_RANK = st.text_input(label="Team2_FIFA_RANK",value=Team2Rank,disabled=True)


Team1GK = rankings[rankings["Team"] == Team1]['GK'].to_list()[0]
Team1Def = rankings[rankings["Team"] == Team1]['DEF'].to_list()[0]
Team1Att = rankings[rankings["Team"] == Team1]['ATT'].to_list()[0]
Team1Off = rankings[rankings["Team"] == Team1]['MID'].to_list()[0]

Team1_Goalkeeper_Score = st.text_input(label="Team1 Goalkeeper Score",value=Team1GK,disabled=True)
Team1_Defense = st.text_input(label="Team1 Defense Score",value=Team1Def,disabled=True)
Team1_Offense = st.text_input(label="Team1 Offense Score",value=Team1Att,disabled=True)
Team1_Midfield = st.text_input(label="Team1 Midfield Score",value=Team1Off,disabled=True)

Team2GK = rankings[rankings["Team"] == Team2]['GK'].to_list()[0]
Team2Def = rankings[rankings["Team"] == Team2]['DEF'].to_list()[0]
Team2Att = rankings[rankings["Team"] == Team2]['ATT'].to_list()[0]
Team2Off = rankings[rankings["Team"] == Team2]['MID'].to_list()[0]

Team2_Goalkeeper_Score = st.text_input(label="Team2 Goalkeeper Score",value=Team2GK,disabled=True)
Team2_Defense = st.text_input(label="Team2 Defense Score",value=Team2Def,disabled=True)
Team2_Offense = st.text_input(label="Team2 Offense Score",value=Team2Att,disabled=True)
Team2_Midfield = st.text_input(label="Team2 Midfield Score",value=Team2Off,disabled=True)

league = os.path.join(ROOT_DIR,'data',"League_Predictions.pkl")
knockout = os.path.join(ROOT_DIR,'data',"KnockOut_Predictions.pkl")
colNames = os.path.join(ROOT_DIR,'data',"col_names.pkl")

league_model = joblib.load(league)
knockout_model = joblib.load(knockout)
col_names = joblib.load(colNames)

list_value = [[Team1, Team2, Team1_FIFA_RANK, Team2_FIFA_RANK, Team1_Goalkeeper_Score, Team2_Goalkeeper_Score, Team1_Defense, Team1_Offense, Team1_Midfield, Team2_Defense, Team2_Offense, Team2_Midfield]]
df = pd.DataFrame(data=list_value,columns=col_names)
st.write(df)

if st.button("League Match Prediction"):
    Output = league_model.predict(df)[0]
    if Output == 1:
        st.success(Team1+ " Won the match")
    elif Output == 2:
        st.success("Match Drawn")
    elif Output == 0:
        st.success(Team2 + " Won the match")

if st.button("KnockOut Match Prediction"):
    Output = knockout_model.predict(df)[0]
    if Output == 1:
        st.success(Team1+ " Won the match")
    elif Output == 0:
        st.success(Team2 + " Won the match")
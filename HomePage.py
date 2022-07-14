import requests
import streamlit as st
from streamlit_lottie import st_lottie

#Lottie function
def lottieapi(url: str):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

#setting Page parameters
st.set_page_config(initial_sidebar_state="auto",page_title="Purple Projects",page_icon="📈",layout='wide')
#st.markdown("<h1 style='text-align: center; color: white;'>Purple Projects</h1>", unsafe_allow_html=True)

#Title and image
st.title("Purple Projects ")
st.image("data/Photo.png", width=150)

#About Me
st.write("Hi, I am Solomon 👋")
st.write("Currently I have been doing data cleaning, creating deep/machine learning models to predict and to analyze the data for project works, My key skills are Python, Data Science, AI, ML, Selenium with Java, UFT, Tosca, Functional Testing, API using RestAssured.")

#Coding gif/image
codingGif = lottieapi("https://assets9.lottiefiles.com/packages/lf20_w51pcehl.json")
st_lottie(codingGif,key="coding",height=700,width=900)

#Connect with me section
connectMeGif = lottieapi("https://assets4.lottiefiles.com/packages/lf20_85jUo8.json")
contact_icon, contact_link = st.columns((0.05,1))
with contact_icon:
    st_lottie(connectMeGif,key="connect")
with contact_link:
    st.subheader("Connect with Me")

#Github
githubGif = lottieapi("https://assets3.lottiefiles.com/packages/lf20_S6vWEd.json")
github_icon, github_link = st.columns((0.04,1))
with github_icon:
    st_lottie(githubGif,key="github")
with github_link:
    st.write("[Github](https://github.com/SolomonJesurathinam)")

#Linkedin
LinkedinGif = lottieapi("https://assets10.lottiefiles.com/packages/lf20_v7gucnur.json")
linkedin_icon, linkedin_link = st.columns((0.04,1))
with linkedin_icon:
    st_lottie(LinkedinGif ,key="linkedin")
with linkedin_link:
    st.write("[Linkedin](https://www.linkedin.com/in/solomon-jesurathinam-a3a80723/)")

#Email
EmailGif = lottieapi("https://assets7.lottiefiles.com/packages/lf20_u25cckyh.json")
Email_icon, Email_link = st.columns((0.04,1))
with Email_icon:
    st_lottie(EmailGif ,key="email")
with Email_link:
    st.write("solomon258@gmail.com")
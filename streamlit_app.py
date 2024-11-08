import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info("This app builds a machine learning model")

df = pd.read_csv("https://raw.githubusercontent.com/ajit267/Diabetes_Prediction/refs/heads/main/diabetes.csv")
df.head()

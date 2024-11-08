import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info("This app builds a machine learning model")

df = pd.read_csv("https://raw.githubusercontent.com/shsarv/Machine-Learning-Projects/refs/heads/main/Diabetes%20Prediction%20%5BEND%202%20END%5D/dataset/kaggle_diabetes.csv")
df.head()

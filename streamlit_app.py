import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info("This app builds a machine learning model")

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/ajit267/Diabetes_Prediction/refs/heads/main/diabetes.csv")
  df
  st.write('**X**')
  x = df.drop('Outcome',axis=1)
  x
  st.write('**Y**')
  y = df.Outcome
  y

with st.expander('Data Visualization'):
  st.scatter_chart(data=df,x='Age',y='BloodPressure',color='Outcome')

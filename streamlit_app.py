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
  st.scatter_chart(data=df,x='BMI',y='BloodPressure',color='Outcome')

# Data Preparation
with st.sidebar:
  st.header('Input Features')
  Pregnancies = st.slider('No OF Pregnancies',0,18,3)
  Glucose = st.slider('Glucose Level',0,200,121)
  BloodPressure = st.slider('BloodPressure:BP',0,122,69)
  SkinThickness = st.slider('SkinThickness',0,110,21)
  Insulin = st.slider('Insulin',0,744,80)
  BMI = st.slider('BMI',0,80,32)
  DiabetesPedigreeFunction = st.slider('DiabetesPedigreeFunction',0.00,2.45,0.47)
  Age = st.slider('Age',21,81,33)

# Create a Data Frame for input Features
data = {'Pregnancies':Pregnancies,
        'Glucose':Glucose,
        'BloodPressure':BloodPressure,
        'SkinThickness':SkinThickness,
        'Insulin':Insulin,
        'BMI':BMI,
        'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
        'Age':Age
}
input_df = pd.DataFrame(data, index=[0])
input_df











  


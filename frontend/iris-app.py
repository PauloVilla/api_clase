import streamlit as st
import pandas as pd
import requests
import json

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sl': sepal_length,
            'sw': sepal_width,
            'pl': petal_length,
            'pw': petal_width}
    features = pd.DataFrame(data, index=[0])
    return data, features


data, df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

if st.button("Predict"):
    url = "http://localhost:8000/api/v1/classify"
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("GET", url, headers=headers, data=json.dumps(data))
    prediction = json.loads(response.text)["Iris flower"]
    st.subheader('Prediction')
    st.write(prediction)

# docker build -t backend-api-aris:v1 .
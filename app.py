import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle


# Load the scaler and model from pickle files
with open('scaled_data.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pkl', 'rb') as f:
    mlp = pickle.load(f)

columns = ['mean radius', 'mean perimeter', 'mean area',
           'mean concavity', 'mean concave points', 'worst radius', 'worst perimeter',
           'worst area', 'worst concavity',
           'worst concave points']


# Function to get user input
def get_user_input():
    user_data = {}
    user_data['mean radius'] = st.text_input('Mean Radius', '0.0')
    user_data['mean perimeter'] = st.text_input('Mean Perimeter', '0.0')
    user_data['mean area'] = st.text_input('Mean Area', '0.0')
    user_data['mean concavity'] = st.text_input('Mean Concavity', '0.0')
    user_data['mean concave points'] = st.text_input('Mean Concave Points', '0.0')
    user_data['worst radius'] = st.text_input('Worst Radius', '0.0')
    user_data['worst perimeter'] = st.text_input('Worst Perimeter', '0.0')
    user_data['worst area'] = st.text_input('Worst Area', '0.0')
    user_data['worst concavity'] = st.text_input('Worst Concavity', '0.0')
    user_data['worst concave points'] = st.text_input('Worst Concave Points', '0.0')

    user_data = {key: float(value) for key, value in user_data.items()}
    return pd.DataFrame([user_data])

# Streamlit app
st.title("Breast Cancer Prediction")

# Get user input
input_data = get_user_input()

# Standardize the input data
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = mlp.predict(input_data_scaled)
    prediction_proba = mlp.predict_proba(input_data_scaled)

    # Output the prediction
    if prediction[0] == 0:
        st.write("Prediction: Malignant")
    else:
        st.write("Prediction: Benign")
    
    st.write("Prediction Probabilities: ", prediction_proba)

# Display feature input values
st.write("Input Values:", input_data)
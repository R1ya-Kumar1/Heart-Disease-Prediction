import streamlit as st
import numpy as np
import pickle

# Load the saved model
model = pickle.load(open(r'C:\Users\Hi\Desktop\Workspace\College\Project\NIT Jamshedpur 2025\models_saved\heart_disease_model.sav', 'rb'))

# Title of the web app
st.title("Heart Disease Prediction Web App")

# Input fields for user data
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

# Prediction button
if st.button("Predict"):
    # Convert inputs into numpy array
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    if prediction[0] == 0:
        st.success("✅ The Person does NOT have Heart Disease")
    else:
        st.error("⚠️ The Person HAS Heart Disease")

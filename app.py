import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')


st.title("Heart Stroke Prediction by Shresth")
st.markdown("This app predicts the likelihood of a heart stroke based on user input. Provide the following details ")
age = st.slider('Age' , 18 , 100 , 40)
sex = st.selectbox("SEX",['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0,1])
resting_ecg = st.selectbox("Resting Electrocardiographic Results", ['Normal', 'ST', 'LVH'])
max_hr =st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ['Y', 'N'])
old_peak = st.slider("Old Peak(ST Depression)", 0.0 ,6.0 ,1.0)
st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

if st.button("Predict"):
    raw_input  = {
        'age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': old_peak,
        'Sex_'+ sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    input_data = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected_columns]
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    
    if prediction == 1:
        st.error('⚠️ High risk of heart stroke detected! Please consult a doctor immediately.')
    else:
        st.success('✅ Low risk of heart stroke detected. Keep maintaining a healthy lifestyle!')
    

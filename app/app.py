import streamlit as st
import pandas as pd
import joblib
import os

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "models", "model.pkl")
    return joblib.load(model_path)

@st.cache_resource
def load_preprocessor():
    preprocessor_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "models", "preprocessor.pkl")
    return joblib.load(preprocessor_path)

def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("Heart Disease Prediction App")
    st.write("Enter the patient's details below to predict the presence of heart disease:")

    age = st.number_input("Age", min_value=1, max_value=110, value=30)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox(
        "Chest Pain Type:\n0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic",
        [0, 1, 2, 3]
    )
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=200, value=120)
    chol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
    restecg = st.selectbox(
        "Resting ECG Results:\n0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy",
        [0, 1, 2]
    )
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment:\n0 = Flat, 1 = Up, 2 = Down", [0, 1, 2])

    sex_map = {0: 'F', 1: 'M'}
    cp_map = {0: 'TA', 1: 'ATA', 2: 'NAP', 3: 'ASY'}
    restecg_map = {0: 'Normal', 1: 'ST', 2: 'LVH'}
    exang_map = {0: 'N', 1: 'Y'}
    slope_map = {0: 'Flat', 1: 'Up', 2: 'Down'}

    input_data = pd.DataFrame({
        "Age": [age],
        "RestingBP": [trestbps],
        "Cholesterol": [chol],
        "FastingBS": [fbs],
        "MaxHR": [thalach],
        "Oldpeak": [oldpeak],
        "Sex": [sex_map[sex]],               # Original categories (string)
        "ChestPainType": [cp_map[cp]],       # Original categories (string)
        "RestingECG": [restecg_map[restecg]],# Original categories (string)
        "ExerciseAngina": [exang_map[exang]],# Original categories (string)
        "ST_Slope": [slope_map[slope]]                  # Numeric (int)
    })

    print(input_data)

    if st.button("Predict"):
        model = load_model()
        preprocessor = load_preprocessor()

        input_transformed = preprocessor.transform(input_data)

        prediction = predict(model, input_transformed)

        if prediction[0] == 1:
            st.markdown(
                "<p style='color:red; font-size:20px; font-weight:bold;'>The model predicts the patient HAS heart disease.</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<p style='color:green; font-size:20px; font-weight:bold;'>The model predicts the patient does NOT have heart disease.</p>",
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()

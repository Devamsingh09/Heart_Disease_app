import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and feature matrix
def load_model_and_X(model_file, X_file):
    with open(model_file, 'rb') as f:
        RF = pickle.load(f)
    X = pd.read_csv(X_file)  # Adjust if your feature matrix is in a different format
    return RF, X

# Prediction function for heart disease
def predict_heart_disease(age, sex, resting_bp, cholesterol, fasting_bs, max_hr, exercise_angina, oldpeak, st_slope, lvh, normal, asy, ata, nap, X, RF):
    x = np.zeros(len(X.columns))
    
    # Assign values to the features
    x[0] = age
    x[1] = sex
    x[2] = resting_bp
    x[3] = cholesterol
    x[4] = fasting_bs
    x[5] = max_hr
    x[6] = exercise_angina
    x[7] = oldpeak
    x[8] = st_slope
    x[9] = lvh
    x[10] = normal
    x[11] = asy
    x[12] = ata
    x[13] = nap
    
    # Predict whether the patient has heart disease
    return RF.predict([x])[0]

# Main function for the Streamlit app
def main():
    st.title("Heart Disease Prediction")

    # Display an image
    st.image('img.jpeg', use_column_width=True)

    st.markdown("<h2 style='text-align: center; color: grey;'>Enter Patient Details</h2>", unsafe_allow_html=True)

    # Use columns to organize input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=50)
        sex = st.selectbox('Sex', ['Female', 'Male'])  # Descriptive labels
        resting_bp = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
        cholesterol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)

    with col2:
        fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])  # Descriptive labels
        max_hr = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])  # Descriptive labels
        oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)

    with col3:
        st_slope = st.selectbox('ST Slope', ['Upsloping', 'Flat', 'Downsloping'])  # Example values; adjust as per your data
        lvh = st.selectbox('Left Ventricular Hypertrophy (LVH)', ['No', 'Yes'])
        normal = st.selectbox('Normal', ['No', 'Yes'])
        asy = st.selectbox('Asymptomatic', ['No', 'Yes'])
        ata = st.selectbox('Atypical Angina (ATA)', ['No', 'Yes'])
        nap = st.selectbox('Non-Anginal Pain (NAP)', ['No', 'Yes'])

    # Map descriptive labels back to numeric values expected by the model
    sex = 0 if sex == 'Female' else 1
    fasting_bs = 0 if fasting_bs == 'No' else 1
    exercise_angina = 0 if exercise_angina == 'No' else 1
    st_slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    st_slope = st_slope_map[st_slope]
    lvh = 0 if lvh == 'No' else 1
    normal = 0 if normal == 'No' else 1
    asy = 0 if asy == 'No' else 1
    ata = 0 if ata == 'No' else 1
    nap = 0 if nap == 'No' else 1

    # Load model and feature matrix
    RF, X = load_model_and_X("model.pkl", "input_dataframe.csv")

    if st.button('Predict'):
        with st.spinner('Calculating...'):
            result = predict_heart_disease(age, sex, resting_bp, cholesterol, fasting_bs, max_hr, exercise_angina, oldpeak, st_slope, lvh, normal, asy, ata, nap, X, RF)
        st.success(f'Heart Disease Prediction: {"Positive" if result == 1 else "Negative"}')

if __name__ == '__main__':
    main()

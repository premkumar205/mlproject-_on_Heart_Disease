import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Heart Disease Predictor")

st.title("🏥 Heart Disease Prediction App")
st.write("Enter patient information below to check heart disease risk")

# Load trained model

model = joblib.load("logistic_regression_model.pkl")

# ---- USER INPUTS ----

age = st.number_input("Age", 20, 100, 45)

sex_option = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_option == "Male" else 0

cp_option = st.selectbox(
"Chest Pain Type",
["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)

cp_map = {
"Typical Angina":0,
"Atypical Angina":1,
"Non-anginal Pain":2,
"Asymptomatic":3
}

cp = cp_map[cp_option]

trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

chol = st.number_input("Cholesterol", 100, 400, 200)

fbs_option = st.selectbox(
"Fasting Blood Sugar",
["Less than 120 mg/dl", "Greater than 120 mg/dl"]
)

fbs = 1 if fbs_option == "Greater than 120 mg/dl" else 0

restecg_option = st.selectbox(
"Rest ECG Result",
["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
)

restecg_map = {
"Normal":0,
"ST-T wave abnormality":1,
"Left ventricular hypertrophy":2
}

restecg = restecg_map[restecg_option]

thalach = st.number_input("Maximum Heart Rate", 70, 210, 150)

exang_option = st.selectbox(
"Exercise Induced Angina",
["No", "Yes"]
)

exang = 1 if exang_option == "Yes" else 0

oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)

slope_option = st.selectbox(
"Slope of ST Segment",
["Upsloping", "Flat", "Downsloping"]
)

slope_map = {
"Upsloping":0,
"Flat":1,
"Downsloping":2
}

slope = slope_map[slope_option]

ca = st.selectbox("Number of Major Vessels", [0,1,2,3])

thal_option = st.selectbox(
"Thalassemia",
["Normal", "Fixed Defect", "Reversible Defect"]
)

thal_map = {
"Normal":1,
"Fixed Defect":2,
"Reversible Defect":3
}

thal = thal_map[thal_option]

# ---- PREDICTION ----

if st.button("Predict Heart Disease Risk"):

input_data = pd.Data([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]],
                          columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0]

if prediction == 1:
    st.error("⚠️ High Risk of Heart Disease")
else:
    st.success("✅ Low Risk of Heart Disease")

st.write("Prediction Confidence:", round(max(probability)*100,2), "%")



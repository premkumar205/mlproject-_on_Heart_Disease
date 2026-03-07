import streamlit as st
import joblib
import pandas as pd

# Page configuration

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.title("🏥 Heart Disease Prediction App")
st.markdown("Provide patient details below to predict heart disease risk.")
st.markdown("---")

# Load model

model = joblib.load("logistic_regression_model.pkl")

user_input = {}

col1, col2 = st.columns(2)

# LEFT SIDE

with col1:
st.subheader("Demographic & Basic Health")

```
user_input["age"] = st.number_input("Age", 20, 100, 45)

sex = st.selectbox("Sex", ["Male", "Female"])
user_input["sex"] = 1 if sex == "Male" else 0

cp = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

user_input["cp"] = cp_map[cp]

user_input["trestbps"] = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    80, 200, 120
)

user_input["chol"] = st.number_input(
    "Cholesterol (mg/dl)",
    100, 400, 200
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    ["Less than 120 mg/dl", "Greater than 120 mg/dl"]
)

user_input["fbs"] = 1 if fbs == "Greater than 120 mg/dl" else 0

restecg = st.selectbox(
    "Resting ECG Result",
    [
        "Normal",
        "ST-T wave abnormality",
        "Left ventricular hypertrophy"
    ]
)

restecg_map = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2
}

user_input["restecg"] = restecg_map[restecg]
```

# RIGHT SIDE

with col2:
st.subheader("Additional Health Metrics")

```
user_input["thalach"] = st.number_input(
    "Maximum Heart Rate Achieved",
    70, 210, 150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

user_input["exang"] = 1 if exang == "Yes" else 0

user_input["oldpeak"] = st.number_input(
    "Oldpeak (ST Depression)",
    0.0, 6.0, 1.0
)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping", "Flat", "Downsloping"]
)

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

user_input["slope"] = slope_map[slope]

user_input["ca"] = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thalassemia",
    ["Normal", "Fixed Defect", "Reversible Defect"]
)

thal_map = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

user_input["thal"] = thal_map[thal]
```

st.markdown("---")

# Prediction

if st.button("🔍 Predict Heart Disease Risk"):

```
input_df = pd.DataFrame([user_input])

prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

st.subheader("📊 Prediction Result")

if prediction == 1:
    st.error("⚠️ HIGH RISK of Heart Disease")
else:
    st.success("✅ LOW RISK of Heart Disease")

st.metric("Confidence", f"{max(prediction_proba)*100:.2f}%")

prob_df = pd.DataFrame({
    "Class": ["No Disease", "Disease"],
    "Probability": prediction_proba
})

st.subheader("📈 Probability Breakdown")
st.bar_chart(prob_df.set_index("Class"))

st.subheader("📝 Patient Input Summary")
st.dataframe(input_df.T)
```

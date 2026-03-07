import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.title("🏥 Heart Disease Prediction App")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("logistic_regression_model.pkl")

model = load_model()

# Feature names (same order used during training)
feature_names = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

st.sidebar.header("📋 Patient Information")

user_input = {}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographic & Health Metrics")

    user_input["age"] = st.number_input("Age", 20, 100, 50)
    user_input["sex"] = st.selectbox("Sex", [0,1])
    user_input["cp"] = st.selectbox("Chest Pain Type", [0,1,2,3])
    user_input["trestbps"] = st.number_input("Resting Blood Pressure", 80,200,120)
    user_input["chol"] = st.number_input("Cholesterol",100,400,200)
    user_input["fbs"] = st.selectbox("Fasting Blood Sugar", [0,1])
    user_input["restecg"] = st.selectbox("Rest ECG", [0,1,2])

with col2:
    st.subheader("Additional Metrics")

    user_input["thalach"] = st.number_input("Max Heart Rate",70,210,150)
    user_input["exang"] = st.selectbox("Exercise Induced Angina", [0,1])
    user_input["oldpeak"] = st.number_input("Oldpeak",0.0,6.0,1.0)
    user_input["slope"] = st.selectbox("Slope", [0,1,2])
    user_input["ca"] = st.selectbox("CA", [0,1,2,3,4])
    user_input["thal"] = st.selectbox("Thal", [0,1,2,3])

st.markdown("---")

if st.button("🔍 Predict", use_container_width=True):

    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("📊 Prediction Result")

    col1,col2,col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("⚠️ HIGH RISK")
        else:
            st.success("✅ LOW RISK")

    with col2:
        st.metric("Confidence", f"{max(prediction_proba)*100:.2f}%")

    with col3:
        st.metric("Prediction Class", prediction)

    st.subheader("📈 Probability Breakdown")

    prob_df = pd.DataFrame({
        "Class":["No Disease","Disease"],
        "Probability":prediction_proba
    })

    st.bar_chart(prob_df.set_index("Class"))

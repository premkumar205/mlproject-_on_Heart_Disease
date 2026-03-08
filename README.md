# ❤️ Heart Disease Prediction using Machine Learning
## live link: https://mlprojectheart.streamlit.app/

## 📌 Project Overview
Heart disease is one of the leading causes of death worldwide. Early prediction can help doctors take preventive actions and improve patient outcomes.

This project uses Machine Learning algorithms to predict whether a person is likely to have heart disease based on several medical attributes such as age, cholesterol level, blood pressure, and chest pain type.

The model is trained on a Heart Disease dataset and classifies patients into:
- 0 → No Heart Disease
- 1 → Heart Disease

---

## 🎯 Objective
The objective of this project is to build a machine learning model that predicts the risk of heart disease using medical data.

Goals:
- Analyze medical data
- Train a classification model
- Predict heart disease probability
- Identify important health features affecting heart disease

---

## 📊 Dataset

The dataset contains medical information about patients.

Features included:

| Feature | Description |
|--------|-------------|
| age | Age of the patient |
| sex | Gender (1 = Male, 0 = Female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Cholesterol level |
| fbs | Fasting blood sugar |
| restecg | ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |
| target | Heart disease (1 = Yes, 0 = No) |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## ⚙️ Machine Learning Workflow

1. **Data Collection**
   - Load the heart disease dataset.

2. **Data Preprocessing**
   - Handle missing values
   - Feature selection
   - Train-test split

3. **Model Training**
   - Logistic Regression
   - Random Forest
   - Decision Tree

4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix

5. **Prediction**
   - Predict whether a patient has heart disease.

---

## 📂 Project Structure

```
mlproject_on_Heart_Disease
│
├── heart.csv
├── heart_disease_prediction.ipynb
├── model.pkl
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```
git clone https://github.com/premkumar205/mlproject-_on_Heart_Disease.git
```

### 2️⃣ Install Dependencies
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3️⃣ Run the Notebook
Open the Jupyter Notebook and run all cells.

---

## 🚀 Future Improvements

- Deploy using Streamlit or Flask
- Improve model accuracy using advanced algorithms
- Create a web interface for prediction
- Use larger medical datasets

---

## 👨‍💻 Author

Chilkamarri Prem Kumar  
B.Tech – Artificial Intelligence & Data Science

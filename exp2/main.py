import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv("heart.csv")
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


joblib.dump(model, "heart_model.pkl")

st.title("💓 Heart Disease Prediction by Prajwal")

st.write("Enter patient data to predict the risk of heart disease.")

age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

sex_num = 1 if sex == "Male" else 0

input_data = pd.DataFrame([[age, sex_num, cp, trestbps, chol, fbs, restecg, thalach,
exang, oldpeak, slope, ca, thal]],
columns=X.columns)

if st.button("Predict"):

prediction = model.predict(input_data)[0]

probability = model.predict_proba(input_data)[0][1]

if prediction == 1:
st.error(f"⚠️ High risk of heart disease (Confidence: {probability:.2%})")

else:
st.success(f"✅ Low risk of heart disease (Confidence: {probability:.2%})")
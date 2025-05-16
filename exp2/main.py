import streamlit as st
#- Imports Streamlit, the library used to create web apps with simple Python scripts.
import pandas as pd
#- Imports Pandas, which is used for handling tabular data (like reading and manipulating datasets).
from sklearn.linear_model import LogisticRegression
#- Imports LogisticRegression from scikit-learn, which is used for building the logistic regression model.
from sklearn.model_selection import train_test_split
#- Imports train_test_split for splitting the dataset into training and testing sets.
from sklearn.metrics import accuracy_score
#- Imports accuracy_score for evaluating the model's performance.  
# from sklearn.datasets import load_heart_disease  # Replace with custom dataset if needed
import joblib
#- Imports joblib for saving the trained model to a file.

# Load dataset
df = pd.read_csv("heart.csv")  # Download from https://www.kaggle.com/datasets/ronitf/heart-disease-uci
#Reads a CSV file containing heart health data from disk into a DataFrame.

X = df.drop('target', axis=1)
#- Drops the target column from the DataFrame to create the feature set.
y = df['target']
#- Extracts the target variable (heart disease presence) from the DataFrame.

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#- Splits the dataset into training and testing sets, with 80% for training and 20% for testing.

# Train the model
model = LogisticRegression(max_iter=1000)
#- Initializes the Logistic Regression model with a maximum of 1000 iterations for convergence.
model.fit(X_train, y_train)
#- Fits the model to the training data.

# Save model (optional)
joblib.dump(model, "heart_model.pkl")
#- Saves the trained model to a file named "heart_model.pkl".

# Streamlit UI
st.title("💓 Heart Disease Prediction by Prajwal")
#- Sets the title of the web app.
st.write("Enter patient data to predict the risk of heart disease.")
#- Displays a description of the app.    

# User input fields
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

# Convert sex to numerical
sex_num = 1 if sex == "Male" else 0
#Converts "Male"/"Female" to 1 or 0, since ML models need numerical input.

# Predict
input_data = pd.DataFrame([[age, sex_num, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]],
                          columns=X.columns)
#Combines all input features into a single data row that matches the training data format.

if st.button("Predict"):
#- Creates a button that, when clicked, triggers the prediction process.
    prediction = model.predict(input_data)[0]
    #- Uses the trained model to predict the target variable based on the input data.
    probability = model.predict_proba(input_data)[0][1]
    #- Calculates the probability of heart disease using the model's predict_proba method.

    if prediction == 1:
        # If the model predicts heart disease (1)
        st.error(f"⚠️ High risk of heart disease (Confidence: {probability:.2%})")
    #Displays an error message indicating a high risk of heart disease.
    else:
        st.success(f"✅ Low risk of heart disease (Confidence: {probability:.2%})")
    #Displays a success message indicating a low risk of heart disease.
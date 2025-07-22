import streamlit as st
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.title("ML Model Playground 🚀")
st.write("Adjust sliders to input feature values and select ML model to predict.")

# Load Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Sidebar - Model Selection
model_name = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "SVM"])

# Sidebar - Feature Input Sliders
def user_input_features():
    sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()))
    sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()))
    petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()))
    petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()))
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Model Setup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def get_model(name):
    if name == "Logistic Regression":
        model = LogisticRegression()
    elif name == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = SVC()
    return model

model = get_model(model_name)
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None

st.subheader("Prediction")
st.write(f"Predicted Class: **{target_names[prediction[0]]}**")

if prediction_proba is not None:
    st.subheader("Prediction Probability")
    st.write(prediction_proba)

# Model Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.subheader("Model Accuracy")
st.write(f"{acc * 100:.2f}%")


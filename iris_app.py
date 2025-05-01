import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load("knn_model.pkl")

st.title("🌸 Iris Flower Classifier 🌸")

# Sliders for input features
sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sw = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
pw = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict on button click
if st.button("Predict Species"):
    prediction = model.predict([[sl, sw, pl, pw]])
    species = ["Setosa", "Versicolor", "Virginica"][prediction[0]]
    st.success(f"The model predicts: **{species}** 🌼")

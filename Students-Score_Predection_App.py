# student_score_app.py
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title("🎓 Student Score Predictor")
st.write("Enter the number of study hours to predict student scores.")

# Training Data
data = {
    "Hours": [1, 2, 3, 4, 5, 6],
    "Marks": [20, 35, 50, 65, 80, 95]
}
df = pd.DataFrame(data)

# Train the model
X = df[["Hours"]]
y = df["Marks"]
model = LinearRegression()
model.fit(X, y)

# User input
hours = st.number_input("🕒 Enter Hours Studied:", min_value=0.0, max_value=12.0, step=0.5)

# Predict
if st.button("🔮 Predict Score"):
    prediction = model.predict([[hours]])
    st.success(f"✅ Predicted Score: {prediction[0]:.2f} / 100")

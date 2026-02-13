import streamlit as st
import numpy as np
import pickle

# Load model
with open("New_XGB_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler if used
try:
    with open("New_scalar.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

st.title("Malnutrition Prediction System")
st.write("Enter child details to predict malnutrition status.")

# ------------------ User Inputs ------------------

#Age = st.number_input("Age (months)", min_value=6.0, max_value=59.0, value=12.0)
#Weight_kg = st.number_input("Weight (kg)", min_value=6.0, max_value=20.0, value=8.0)
#Height_cm = st.number_input("Height (cm)", min_value=60.0, max_value=120.0, value=75.0)
#Hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=7.0, max_value=14.0, value=10.0)
#Meals_per_day = st.number_input("Meals per day", min_value=1.0, max_value=4.0, value=3.0)
Age = st.number_input(
    "Age (months)",
    min_value=6.0,
    max_value=59.0,
    value=12.0,
    step=0.1,
    format="%.1f"
)

Weight_kg = st.number_input(
    "Weight (kg)",
    min_value=6.0,
    max_value=20.0,
    value=8.0,
    step=0.1,
    format="%.1f"
)

Height_cm = st.number_input(
    "Height (cm)",
    min_value=60.0,
    max_value=120.0,
    value=75.0,
    step=0.1,
    format="%.1f"
)

Hemoglobin = st.number_input(
    "Hemoglobin (g/dL)",
    min_value=7.0,
    max_value=14.0,
    value=10.0,
    step=0.1,
    format="%.1f"
)

Meals_per_day = st.selectbox("Meals per day", [1, 2, 3, 4])

gender = st.selectbox("Gender", ["Female", "Male"])
Gender = 1 if gender == "Male" else 0

# Combine inputs (MUST match training feature order)
input_data = np.array([[ 
    Age,
    Gender,
    Weight_kg,
    Height_cm,
    Hemoglobin,
    Meals_per_day
]])

# Apply scaling only if scaler exists
if scaler is not None:
    input_data = scaler.transform(input_data)

# ------------------ Prediction ------------------

if st.button("Predict Malnutrition Status"):

    prediction = model.predict(input_data)
    pred_class = int(prediction[0])   # No rounding needed

    status_map = {
        0: "Normal",
        1: "Moderate",
        2: "Severe"
    }

    result = status_map.get(pred_class, "Unknown")

    # Display result
    if result == "Normal":
        st.success(f"Estimated Malnutrition Status: {result}")
    elif result == "Moderate":
        st.warning(f"Estimated Malnutrition Status: {result}")
    else:
        st.error(f"Estimated Malnutrition Status: {result}")

    # Show prediction probability (if available)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0]
        prob_map = {
            status_map[i]: round(prob[i] * 100, 2)
            for i in range(len(prob))
        }
        st.info(f"Prediction Confidence (%): {prob_map}")


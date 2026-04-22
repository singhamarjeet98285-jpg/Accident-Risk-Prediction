# =====================================
# IMPORT LIBRARIES
# =====================================
import streamlit as st
import pandas as pd
import pickle

# =====================================
# LOAD MODEL SAFELY
# =====================================
try:
    model = pickle.load(open("model.pkl", "rb"))
    le_dict = pickle.load(open("encoders.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# =====================================
# UI
# =====================================
st.set_page_config(page_title="Accident Risk", layout="centered")

st.title("🚨 Accident Risk Prediction")
st.write("Fill the details below")

# =====================================
# CHECK REQUIRED KEYS
# =====================================
required_cols = [
    "State Name", "City Name", "Weather Conditions",
    "Road Type", "Road Condition", "Lighting Conditions"
]

for col in required_cols:
    if col not in le_dict:
        st.error(f"Missing column in encoder: {col}")
        st.stop()

# =====================================
# INPUTS
# =====================================
state = st.selectbox("State", le_dict["State Name"].classes_)
city = st.selectbox("City", le_dict["City Name"].classes_)
weather = st.selectbox("Weather", le_dict["Weather Conditions"].classes_)
road_type = st.selectbox("Road Type", le_dict["Road Type"].classes_)
road_condition = st.selectbox("Road Condition", le_dict["Road Condition"].classes_)
lighting = st.selectbox("Lighting", le_dict["Lighting Conditions"].classes_)

vehicles = st.number_input("Vehicles", 1, 20, 2)
casualties = st.number_input("Casualties", 0, 20, 1)
fatalities = st.number_input("Fatalities", 0, 10, 0)
speed = st.number_input("Speed (km/h)", 20, 200, 60)
age = st.number_input("Driver Age", 18, 80, 30)

# =====================================
# PREDICT
# =====================================
if st.button("Predict Risk"):

    try:
        input_data = {
            "State Name": le_dict["State Name"].transform([state])[0],
            "City Name": le_dict["City Name"].transform([city])[0],
            "Weather Conditions": le_dict["Weather Conditions"].transform([weather])[0],
            "Road Type": le_dict["Road Type"].transform([road_type])[0],
            "Road Condition": le_dict["Road Condition"].transform([road_condition])[0],
            "Lighting Conditions": le_dict["Lighting Conditions"].transform([lighting])[0],
            "Number of Vehicles Involved": vehicles,
            "Number of Casualties": casualties,
            "Number of Fatalities": fatalities,
            "Speed Limit (km/h)": speed,
            "Driver Age": age
        }

        input_df = pd.DataFrame([input_data])

        # FIX: match training columns safely
        if hasattr(model, "feature_names_in_"):
            expected_cols = model.feature_names_in_
            for col in expected_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[expected_cols]

        # Predict
        pred = model.predict(input_df)

        # Handle label type safely
        risk_map = {0: "Low", 1: "Medium", 2: "High"}

        if isinstance(pred[0], (int, float)):
            result = risk_map.get(pred[0], pred[0])
        else:
            result = pred[0]

        st.success(f"🚨 Predicted Risk: {result}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# =====================================
# IMPORT LIBRARIES
# =====================================
import streamlit as st
import pandas as pd
import pickle

# =====================================
# LOAD MODEL + ENCODERS
# =====================================
model = pickle.load(open("model.pkl", "rb"))
le_dict = pickle.load(open("encoders.pkl", "rb"))

# =====================================
# UI DESIGN
# =====================================
st.set_page_config(page_title="Accident Risk Prediction", layout="centered")

st.title("🚨 Accident Risk Prediction System")
st.markdown("### Enter details to predict accident risk")

# =====================================
# INPUT FIELDS
# =====================================
state = st.selectbox("State Name", le_dict["State Name"].classes_)
city = st.selectbox("City Name", le_dict["City Name"].classes_)
weather = st.selectbox("Weather Conditions", le_dict["Weather Conditions"].classes_)
road_type = st.selectbox("Road Type", le_dict["Road Type"].classes_)
road_condition = st.selectbox("Road Condition", le_dict["Road Condition"].classes_)
lighting = st.selectbox("Lighting Conditions", le_dict["Lighting Conditions"].classes_)

vehicles = st.number_input("Number of Vehicles Involved", 1, 20, 2)
casualties = st.number_input("Number of Casualties", 0, 20, 1)
fatalities = st.number_input("Number of Fatalities", 0, 10, 0)
speed = st.number_input("Speed Limit (km/h)", 20, 200, 60)
age = st.number_input("Driver Age", 18, 80, 30)

# =====================================
# PREDICTION
# =====================================
if st.button("Predict Risk"):

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

    # Match training columns
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]

    # Prediction
    pred = model.predict(input_df)

    # Risk mapping
    risk_map = {0: "Low", 1: "Medium", 2: "High"}

    st.success(f"🚨 Predicted Accident Risk: {risk_map[pred[0]]}")

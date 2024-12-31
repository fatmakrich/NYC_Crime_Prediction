import pandas as pd
import folium
from streamlit_folium import st_folium
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from openlocationcode import openlocationcode as olc
import joblib
import numpy as np

# Load datasets
file_path_original = 'original_data.csv'  # Dataset before encoding
file_path_encoded = 'Encoded_data.csv'  # Dataset after encoding
data_original = pd.read_csv(file_path_original)
data_encoded = pd.read_csv(file_path_encoded)

# Load the machine learning models
model1 = joblib.load('lgb_model.joblib')  # LightGBM
model2 = joblib.load('xgb_model.joblib')  # XGBoost
model3 = tf.keras.models.load_model('Neural_Network.h5')  # Neural Network

# Sidebar content
with st.sidebar:
    st.markdown("# NYC Crime Prediction Project")
    st.markdown("""
    This project predicts the type of crime that is most likely to occur in different areas of New York City based on demographics and location.""")
    st.markdown("""
    **Prepared by:**  
    Malek ELMECHI  
    Fatma KRICHEN  
    Nouha BEN HAMADA  
                 
    **Supervised by:**  
    Riadh TEBOURBI  
    """)

# Main content area
with st.container():
    st.title("New York Crime Prediction")
    st.markdown("""
    This application uses machine learning algorithms to estimate the probability of a crime taking place in various locations within *New York City*. This tool provides valuable insights based on several key factors, including:
    - Insights based on age, gender, and ethnicity, as these can influence crime patterns.
    - The day and time you plan to be in a particular area, as crime rates can fluctuate throughout the day.
    - The environment you're in—whether it's a park, subway station, or residential area.
    ### How It Works for You:
    1. *Pick Your Destination*: Use the interactive map to choose the area.
    2. *Fill in Your Details*: Enter personal information such as your age, gender, and race.
    3. *Get Crime Predictions*: The app analyzes your data using a majority vote from three models to estimate the likelihood of various types of crimes happening in that location.
    """)

# Interactive map for region selection
st.subheader("Select Your Region on the Map")
map_center = [40.7128, -74.0060]  # Centered on NYC
crime_map = folium.Map(location=map_center, zoom_start=12)
crime_map.add_child(folium.LatLngPopup())
map_data = st_folium(crime_map, width=700, height=500)

# Retrieve coordinates or set defaults
latitude, longitude = 40.7128, -74.0060
if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
    latitude = map_data["last_clicked"]["lat"]
    longitude = map_data["last_clicked"]["lng"]

st.write(f"Selected Latitude: {latitude}")
st.write(f"Selected Longitude: {longitude}")

# Generate the Open Location Code
location_code = olc.encode(latitude, longitude, codeLength=8)
st.write(f"Generated Location Code: {location_code}")

# Encode LOCATION_CODE
le_location = LabelEncoder()
le_location.fit(data_original['LOCATION_CODE'])
location_code_encoded = (
    le_location.transform([location_code])[0]
    if location_code in le_location.classes_
    else -1  # Default for unseen locations
)

# Collect user inputs
st.subheader("Enter Crime Details")
year = st.number_input("Year", min_value=2000, max_value=2025, value=2023)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
hour = st.number_input("Hour", min_value=0, max_value=23, value=12)
weekday = st.selectbox("Weekday", data_original['weekday'].unique())
addr_pct_cd = st.selectbox("Police District Address", data_original['ADDR_PCT_CD'].unique())
crime_class = st.selectbox("Crime Class", data_original['CRIME_CLASS'].unique())
vic_age_group = st.selectbox("Victim's Age Group", data_original['VIC_AGE_GROUP'].unique())
vic_race = st.selectbox("Victim's Race", data_original['VIC_RACE'].unique())
vic_sex = st.selectbox("Victim's Sex", data_original['VIC_SEX'].unique())

# Prepare input for prediction
input_data = pd.DataFrame([{
    "year": year,
    "month": month,
    "hour": hour,
    "weekday": data_encoded.loc[data_original['weekday'] == weekday, 'weekday'].values[0],
    "ADDR_PCT_CD": data_encoded.loc[data_original['ADDR_PCT_CD'] == addr_pct_cd, 'ADDR_PCT_CD'].values[0],
    "CRIME_CLASS": data_encoded.loc[data_original['CRIME_CLASS'] == crime_class, 'CRIME_CLASS'].values[0],
    "VIC_AGE_GROUP": data_encoded.loc[data_original['VIC_AGE_GROUP'] == vic_age_group, 'VIC_AGE_GROUP'].values[0],
    "VIC_RACE": data_encoded.loc[data_original['VIC_RACE'] == vic_race, 'VIC_RACE'].values[0],
    "VIC_SEX": data_encoded.loc[data_original['VIC_SEX'] == vic_sex, 'VIC_SEX'].values[0],
    "LOCATION_CODE": location_code_encoded
}])

# Prediction logic with majority voting
if st.button("Predict Crime Type"):
    try:
        # Predict using all models
        pred1 = model1.predict(input_data)
        pred2 = model2.predict(input_data)
        pred3 = model3.predict(input_data)

        # Ensure predictions are class indices
        pred1 = np.argmax(pred1, axis=-1)[0] if len(pred1.shape) > 1 else int(pred1)
        pred2 = np.argmax(pred2, axis=-1)[0] if len(pred2.shape) > 1 else int(pred2)
        pred3 = np.argmax(pred3, axis=-1)[0] if len(pred3.shape) > 1 else int(pred3)

        # Majority voting
        predictions = [pred1, pred2, pred3]
        predicted_class_index = max(set(predictions), key=predictions.count)

        # Map prediction index to crime type
        target_mapping = {0: 'PROPERTY', 1: 'PERSONAL', 2: 'SEXUAL', 3: 'DRUGS/ALCOHOL'}
        crime_type = target_mapping.get(predicted_class_index, "Unknown")

        # Display results
        st.subheader("Prediction Results")
        st.write(f"Predicted Crime Type: {crime_type}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

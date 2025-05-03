import time  # Import the time module
import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk

loaded_model = pk.load(open('D:/Work/Machine Learning/Deploying Machine Learning Model/trained_model.sav', 'rb'))

st.header('CAR PRICE PREDICTION MODEL')

cars_data=pd.read_csv('D:/Work/Machine Learning/Deploying Machine Learning Model/CarDetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

car_name = st.selectbox('Select Car Brand', cars_data['name'].unique())
manufactured_year = st.slider('Car Manufactured Year', 1994, 2024)
kms_driven = st.slider('No of KMS Driven', 11, 200000)
fuel_type = st.selectbox('Fuel Type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller_Type', cars_data['seller_type'].unique())
transmission_type = st.selectbox('Transmission_Type', cars_data['transmission'].unique())
car_mileage = st.slider('Car Mileage', 10, 40)
engine_capacity = st.slider('Engine Capacity', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
no_of_seats = st.slider('No of Seats', 5, 10)
owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])



# Your existing code here...

# Define the path to the audio file
audio_file = 'D:/Work/Machine Learning/Deploying Machine Learning Model/money-pickup-2-89563.mp3'

audio_js = f"""
<script>
function playAudio() {{
    var audio = new Audio('{audio_file}');
    audio.play();
}}
</script>
"""

# Display the predict button
if st.button("Predict", key="predict_button", help="Play sound and predict"):
    # Execute JavaScript to play audio
    st.markdown(audio_js, unsafe_allow_html=True)
    st.markdown("<script>playAudio();</script>", unsafe_allow_html=True)

    with st.spinner('Predicting...'):  # Display spinner animation while predicting
        # Mapping categorical values to numerical codes
        owner_mapping = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5}
        fuel_mapping = {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4}
        seller_mapping = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
        transmission_mapping = {'Manual': 1, 'Automatic': 2}
        name_mapping = dict(zip(cars_data['name'].unique(), range(1, len(cars_data['name'].unique()) + 1)))

        # Converting selected values to numerical codes
        owner = owner_mapping.get(owner)
        fuel = fuel_mapping.get(fuel_type)
        seller_type = seller_mapping.get(seller_type)
        transmission = transmission_mapping.get(transmission_type)
        name = name_mapping.get(car_name)

        # Creating DataFrame for prediction
        input_data_model = pd.DataFrame([[name, manufactured_year, kms_driven, fuel, seller_type, transmission, owner, car_mileage, engine_capacity, max_power, no_of_seats]],
                                        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])

        # Simulate a delay to make the spinner visible
        time.sleep(2)  # You can adjust the duration as needed

        car_price = loaded_model.predict(input_data_model)
        st.markdown(f'**Car Price Prediction:** ₹ {car_price[0]:,.2f}')
        
        
        st.balloons()  # Display balloon animation after prediction is complete
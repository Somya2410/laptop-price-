import streamlit as st
import pandas as pd
import pickle

# Load dataset and model
df = pd.read_csv('Laptop_price.csv')
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title='Laptop Price Predictor', layout='centered')
st.title("Laptop Price Predictor")

# User inputs
brand = st.selectbox("Brand", df['Brand'].unique())
processor = st.selectbox("Processor Speed", sorted(df['Processor_Speed'].unique()))
ram = st.slider("RAM Size (in GB)", int(df['RAM_Size'].min()), int(df['RAM_Size'].max()))
storage = st.slider("Storage Capacity (in GB)", int(df['Storage_Capacity'].min()), int(df['Storage_Capacity'].max()))
screen = st.slider("Screen Size (in inches)", float(df['Screen_Size'].min()), float(df['Screen_Size'].max()))
weight = st.slider("Weight (in kg)", float(df['Weight'].min()), float(df['Weight'].max()))

# Prepare input
input_data = pd.DataFrame([[brand, processor, ram, storage, screen, weight]], 
    columns=['Brand', 'Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight'])

# Match data types
for col in input_data.columns:
    input_data[col] = input_data[col].astype(df[col].dtype)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Laptop Price: €{round(prediction, 2)}")

# Footer
st.markdown("---")
st.markdown("**Author**: Somya Nigam")
[LinkedIn Profile](https://www.linkedin.com/in/somya-nigam-789408183/)")

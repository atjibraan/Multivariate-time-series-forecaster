#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime

# Load model and scaler
model = tf.keras.models.load_model('transformer_forecaster.h5')
scaler = joblib.load('scaler.pkl')

# Feature columns (after SHAP-based selection)
feature_cols = [
    'CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'NMHC(GT)',
    'T', 'RH', 'PT08.S2(NMHC)', 'NOx(GT)', 
    'PT08.S3(NOX)', 'PT08.S4(NO2)'
]

st.title("Air Quality Forecasting System")

# Current time display
now = datetime.now()
st.header(f"Current Time: {now.strftime('%Y-%m-%d %H:%M')}")

# Input sliders for last 24 hours
st.sidebar.header("Input Sensor Readings")
input_data = []
for feature in feature_cols:
    input_data.append(st.sidebar.slider(
        feature, 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        help=f"Normalized value of {feature}"
    ))

if st.button("Predict Next Hour"):
    # Create sequence
    input_seq = np.array(input_data).reshape(1, 1, len(feature_cols))
    input_seq = np.repeat(input_seq, 24, axis=1)  # Create 24-hour sequence
    
    # Predict
    prediction = model.predict(input_seq)[0]
    
    # Inverse transform
    dummy = np.zeros((1, len(feature_cols)))
    dummy[:, :4] = prediction.reshape(1, -1)  # First 4 are targets
    pred_inv = scaler.inverse_transform(dummy)[0, :4]
    
    # Display results
    st.subheader("Next Hour Prediction:")
    st.metric("CO", f"{pred_inv[0]:.2f} ppm")
    st.metric("PT08.S1(CO)", f"{pred_inv[1]:.2f}")
    st.metric("Benzene (C6H6)", f"{pred_inv[2]:.2f} µg/m³")
    st.metric("NMHC", f"{pred_inv[3]:.2f} ppm")
    
    # Feature importance visualization
    st.subheader("Most Important Features")
    st.image("shap_summary.png", caption="SHAP Feature Importance")

st.markdown("---")
st.subheader("Model Information")
st.write("Transformer-based forecasting model trained on air quality data")


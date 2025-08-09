import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# --------------------------
# Load Saved Artifacts
# --------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("transformer_forecaster.h5", compile=False)
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_sample_data():
    return np.load("X_train.npy")

model = load_model()
scaler = load_scaler()
sample_data = load_sample_data()

SEQ_LENGTH = 24
TARGET_COLS = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)']
FEATURE_COLS = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
    'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
    'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
    'hour', 'day_of_week', 'month'
]

# --------------------------
# Preprocessing Function
# --------------------------
def preprocess_data(uploaded_file):
    df = pd.read_excel(uploaded_file, header=0, skiprows=[1])
    df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
    df.replace(-200, np.nan, inplace=True)

    # Parse date and timestamp
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['timestamp'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df.drop(['Date', 'Time'], axis=1, inplace=True)

    # Interpolation
    df.interpolate(method='linear', inplace=True)
    df.dropna(inplace=True)

    # Add time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Keep only required features
    feature_cols = [col for col in FEATURE_COLS if col in df.columns]
    return df[['timestamp'] + feature_cols]

# --------------------------
# Prediction Function
# --------------------------
def predict_next(df):
    if len(df) < SEQ_LENGTH:
        st.error(f"Need at least {SEQ_LENGTH} rows to make prediction.")
        return None

    seq_data = df[FEATURE_COLS].values[-SEQ_LENGTH:]
    seq_scaled = scaler.transform(seq_data)
    seq_scaled = np.expand_dims(seq_scaled, axis=0)

    pred_scaled = model.predict(seq_scaled)
    last_row_scaled = seq_scaled[-1]
    
    # Convert back to original scale
    # Only for target columns
    dummy = np.zeros((1, len(FEATURE_COLS)))
    dummy[0, :len(TARGET_COLS)] = pred_scaled[0]
    pred_unscaled = scaler.inverse_transform(dummy)[0][:len(TARGET_COLS)]

    return pred_unscaled

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Air Quality Transformer Forecaster", layout="wide")
st.title("🌬️ Air Quality Time Series Forecaster")
st.markdown("Upload your AirQualityUCI dataset to get **next-hour predictions** for key pollutants.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = preprocess_data(uploaded_file)
        st.success("✅ File loaded and preprocessed successfully.")
        st.write(df.tail(10))

        prediction = predict_next(df)
        if prediction is not None:
            st.subheader("📊 Next Hour Prediction")
            pred_df = pd.DataFrame([prediction], columns=TARGET_COLS)
            st.write(pred_df)

            # Visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(TARGET_COLS, prediction, color='skyblue')
            ax.set_title("Predicted Next Hour Air Quality Values")
            ax.set_ylabel("Value")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Please upload an Excel file to begin.")

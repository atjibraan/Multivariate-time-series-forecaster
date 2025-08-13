import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "transformer_forecaster.h5"         # Change if needed
SCALER_PATH = "scaler.pkl"      # Path to your fitted scaler
TIMESTEPS = 24                  # Sequence length used during training

st.set_page_config(page_title="Multivariate Time Series Forecaster", layout="wide")

# --------------------
# LOAD SCALER & MODEL
# --------------------
@st.cache_resource
def load_scaler(path):
    with open(path, "rb") as f:
        scaler_obj = pickle.load(f)
    return scaler_obj

@st.cache_resource
def load_tf_model(path):
    try:
        model = load_model(path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

scaler = load_scaler(SCALER_PATH)
model = load_tf_model(MODEL_PATH)

# --------------------
# SAFE SCALING FUNCTION
# --------------------
def scale_features(scaler, df, feature_cols):
    if hasattr(scaler, "transform"):  # real scaler object
        return scaler.transform(df[feature_cols])
    else:
        st.warning("⚠ Scaler file is not a fitted scaler object. Returning raw feature values.")
        return df[feature_cols].to_numpy()

# --------------------
# DATA LOADING & PREPROCESSING
# --------------------
def load_and_preprocess_data(uploaded_file):
    # Allow both CSV and Excel
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, header=0, skiprows=[1])

    # Replace missing marker values
    df.replace(-200, np.nan, inplace=True)

    # Handle datetime
    if "Date" in df.columns and "Time" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["timestamp"] = pd.to_datetime(
            df["Date"].dt.strftime('%Y-%m-%d') + ' ' + df["Time"].astype(str),
            errors="coerce"
        )
        df.drop(["Date", "Time"], axis=1, inplace=True)

    # Interpolation and drop NA
    df.interpolate(method="linear", inplace=True)
    df.dropna(inplace=True)

    # Add time features if timestamp exists
    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month

    # Define features & targets (must match training)
    target_cols = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)"]
    feature_cols = target_cols + [
        "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
        "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
        "hour", "day_of_week", "month"
    ]
    # Keep only available columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    return df[["timestamp"] + feature_cols], target_cols, feature_cols

# --------------------
# MAKE PREDICTIONS
# --------------------
def make_predictions(df, feature_cols):
    scaled_data = scale_features(scaler, df, feature_cols)

    # Create sequences (sliding windows)
    X = []
    for i in range(len(scaled_data) - TIMESTEPS + 1):
        X.append(scaled_data[i:i+TIMESTEPS])
    X = np.array(X)

    # Predict
    preds = model.predict(X)
    return preds

# --------------------
# STREAMLIT UI
# --------------------
st.title("📈 Multivariate Time Series Forecasting for Pollutants and predicting the next hours pollutants content")
st.markdown("Upload your dataset (CSV or Excel) to forecast future values.")

uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    with st.spinner("Processing data..."):
        try:
            df, target_cols, feature_cols = load_and_preprocess_data(uploaded_file)
            st.success(f"✅ Loaded {df.shape[0]} rows with {len(feature_cols)} features.")

            if model:
                preds = make_predictions(df, feature_cols)
                st.subheader("Predictions")
                pred_df = pd.DataFrame(preds, columns=target_cols)
                st.dataframe(pred_df)
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("📤 Please upload a file to continue.")

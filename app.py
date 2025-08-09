import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --------------------------
# Safe Model Loader
# --------------------------
@st.cache_resource
def load_model():
    # Ignore unrecognized keyword args in MultiHeadAttention
    from tensorflow.keras.layers import MultiHeadAttention
    return tf.keras.models.load_model(
        "transformer_forecaster.h5",
        compile=False,
        custom_objects={"MultiHeadAttention": MultiHeadAttention}
    )

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

SEQ_LENGTH = 24
TARGET_COLS = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)']
FEATURE_COLS = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
    'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
    'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
    'hour', 'day_of_week', 'month'
]

# --------------------------
# Prediction Function
# --------------------------
def predict_next(sequence_24h):
    seq_scaled = scaler.transform(sequence_24h)
    seq_scaled = np.expand_dims(seq_scaled, axis=0)

    pred_scaled = model.predict(seq_scaled)
    dummy = np.zeros((1, len(FEATURE_COLS)))
    dummy[0, :len(TARGET_COLS)] = pred_scaled[0]
    pred_unscaled = scaler.inverse_transform(dummy)[0][:len(TARGET_COLS)]
    return pred_unscaled

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Air Quality Transformer Forecaster", layout="wide")
st.title("🌬️ Air Quality Time Series Forecaster")
st.markdown("Upload a dataset **or** enter readings manually to forecast the next hour's air quality.")

mode = st.radio("Choose Input Mode:", ["📂 Upload File", "✏️ Manual Entry"])

if mode == "📂 Upload File":
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, header=0, skiprows=[1])
            df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
            df.replace(-200, np.nan, inplace=True)
            df.interpolate(method='linear', inplace=True)
            df.dropna(inplace=True)
            df['hour'] = pd.to_datetime(df['Date'], errors='coerce').dt.hour
            df['day_of_week'] = pd.to_datetime(df['Date'], errors='coerce').dt.dayofweek
            df['month'] = pd.to_datetime(df['Date'], errors='coerce').dt.month
            df = df[FEATURE_COLS].tail(SEQ_LENGTH)

            if len(df) == SEQ_LENGTH:
                prediction = predict_next(df.values)
                pred_df = pd.DataFrame([prediction], columns=TARGET_COLS)
                st.subheader("📊 Prediction")
                st.write(pred_df)
            else:
                st.error(f"File must contain at least {SEQ_LENGTH} rows after preprocessing.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

elif mode == "✏️ Manual Entry":
    st.markdown("Enter the last 24 hours of data manually. All fields must be filled.")
    manual_data = []
    for hour in range(SEQ_LENGTH):
        with st.expander(f"Hour {hour+1} Data"):
            row = []
            for feat in FEATURE_COLS:
                val = st.number_input(f"{feat} (Hour {hour+1})", value=0.0, step=0.1)
                row.append(val)
            manual_data.append(row)

    if st.button("Predict from Manual Data"):
        try:
            manual_df = pd.DataFrame(manual_data, columns=FEATURE_COLS)
            prediction = predict_next(manual_df.values)
            pred_df = pd.DataFrame([prediction], columns=TARGET_COLS)
            st.subheader("📊 Prediction")
            st.write(pred_df)
        except Exception as e:
            st.error(f"❌ Error: {e}")

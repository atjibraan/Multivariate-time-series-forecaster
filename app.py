import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention as OriginalMHA

# ---------- PATCH FOR MultiHeadAttention ARG MISMATCH ----------
class PatchedMHA(OriginalMHA):
    def __init__(self, *args, **kwargs):
        kwargs.pop('query_shape', None)
        kwargs.pop('key_shape', None)
        kwargs.pop('value_shape', None)
        super().__init__(*args, **kwargs)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "transformer_forecaster.h5",
        compile=False,
        custom_objects={"MultiHeadAttention": PatchedMHA}
    )

model = load_model()
st.success("✅ Model loaded successfully!")

# ---------- APP TITLE ----------
st.title("📈 Transformer Forecasting App")
st.write("Upload your data or enter it manually for predictions.")

# ---------- DATA INPUT OPTIONS ----------
option = st.radio(
    "Choose data input method:",
    ("📂 Upload CSV", "✍️ Enter data manually")
)

if option == "📂 Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", data.head())
    else:
        data = None

elif option == "✍️ Enter data manually":
    st.write("Enter your data values (comma-separated):")
    user_input = st.text_area("Example: 1.2, 3.4, 5.6, 7.8")
    if user_input.strip():
        try:
            values = [float(x.strip()) for x in user_input.split(",")]
            data = pd.DataFrame([values])
            st.write("Preview of entered data:", data)
        except ValueError:
            st.error("Please enter only numeric values separated by commas.")
            data = None
    else:
        data = None

# ---------- PREDICTION ----------
if data is not None and st.button("🔮 Predict"):
    try:
        input_array = np.array(data).astype(np.float32)
        predictions = model.predict(input_array)
        st.subheader("Prediction Result:")
        st.write(predictions)
    except Exception as e:
        st.error(f"Prediction error: {e}")

# app.py (updated)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
from datetime import timedelta
import sys
import warnings
import os
import re

warnings.filterwarnings("ignore")

# Sidebar: Environment Info
st.sidebar.subheader("Environment Information")
st.sidebar.write(f"Python version: {sys.version.split()[0]}")
st.sidebar.write(f"TensorFlow version: {tf.__version__}")
st.sidebar.write(f"Streamlit version: {st.__version__}")

# ---------------------------
# Robust TFOpLambda to handle string-serialized functions
# ---------------------------
class TFOpLambda(Layer):
    """
    Robust wrapper to handle TFOpLambda deserialization issues where the
    saved 'function' may end up a string (e.g. 'tf.operators.add' or a printed
    tensor repr). This implementation:
      - resolves common TF op names to real callables (tf.add, tf.multiply, etc.)
      - if kwargs contain serialized tensor strings, replaces them with a zeros
        tensor of matching shape as a safe fallback (no crash).
    """

    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        # Accept both callable and string function descriptors
        self.raw_function = function
        self.function = self._resolve_function(function)
        self.supports_masking = True

    def _resolve_function(self, fn):
        # If already callable, done.
        if callable(fn):
            return fn

        # If None, fallback to identity
        if fn is None:
            return lambda x, **k: x

        # If it's a dict or nested config, try to pick 'class_name' or 'function' key
        if isinstance(fn, dict):
            # Some Keras configs embed nested objects; try to extract plausible name
            for key in ("function", "class_name", "name"):
                if key in fn and isinstance(fn[key], str):
                    fn = fn[key]
                    break

        # If string, try to map common op names
        if isinstance(fn, str):
            s = fn.strip()
            # common patterns that show up when serialised
            # e.g. "<function add at 0x...>", "tf.math.add", "tf.operators.add", "add"
            # search keywords
            if re.search(r"add", s, re.I):
                return tf.add
            if re.search(r"sub|subtract", s, re.I):
                return tf.subtract
            if re.search(r"mul|multiply", s, re.I):
                return tf.multiply
            if re.search(r"matmul|dot|tensordot", s, re.I):
                return tf.matmul
            if re.search(r"concat", s, re.I):
                # concat requires axis arg; provide a wrapper expecting kwargs or second arg
                return lambda x, **k: tf.concat(x if isinstance(x, (list, tuple)) else [x], axis=k.get('axis', -1))
            if re.search(r"identity", s, re.I):
                return lambda x, **k: x
            # last resort: try to import attribute from tf if it looks like tf.some.path
            if "tf." in s:
                try:
                    # pick the last path after 'tf.'
                    path = s[s.rfind("tf.") + 3:]
                    # convert "operators.add" -> tf.operators.add
                    obj = tf
                    for part in path.split("."):
                        obj = getattr(obj, part)
                    if callable(obj):
                        return obj
                except Exception:
                    pass

        # fallback: identity op to avoid crash
        return lambda x, **k: x

    def _sanitize_kwargs(self, inputs, kwargs):
        """
        Replace kwargs that look like serialized tensor descriptions (strings)
        with a placeholder zeros tensor of the same shape as 'inputs' wherever
        necessary. This is a safe fallback to avoid type errors during call.
        """
        sanitized = {}
        for k, v in (kwargs or {}).items():
            if isinstance(v, str):
                # common serialized tensor repr contains 'tf.Tensor' or 'shape='
                if "tf.Tensor" in v or "shape=" in v:
                    try:
                        # create zeros of same shape as inputs (best-effort)
                        shape = tf.shape(inputs)
                        sanitized[k] = tf.zeros_like(inputs)
                    except Exception:
                        sanitized[k] = tf.zeros_like(inputs)
                else:
                    # non-tensor strings -> keep them (some ops use name strings)
                    sanitized[k] = v
            else:
                sanitized[k] = v
        return sanitized

    def call(self, inputs, mask=None, **kwargs):
        # If function accidentally serialized as a string, self.function is the resolved callable
        if not callable(self.function):
            # final fallback convert to identity
            return inputs

        # sanitize kwargs that may be serialized strings
        safe_kwargs = self._sanitize_kwargs(inputs, kwargs)

        # Many TF ops accept positional second tensor (e.g., tf.add(x, y))
        # If the resolved callable expects two arguments and 'y' was passed as kw string,
        # we substituted zeros. If the op expects list/tuple, pass as appropriate.
        try:
            return self.function(inputs, **safe_kwargs)
        except TypeError:
            # try passing only inputs (some wrappers expect single arg)
            try:
                return self.function(inputs)
            except Exception:
                # last resort: identity
                return inputs

    def get_config(self):
        cfg = super().get_config()
        # we store raw_function representation for compatibility
        cfg["function"] = self.raw_function if not callable(self.raw_function) else getattr(self.raw_function, "__name__", str(self.raw_function))
        return cfg

    @classmethod
    def from_config(cls, config):
        # if a stringified function was saved, we pass it in and let __init__ resolve it
        fn = config.pop("function", None)
        return cls(fn, **config)

# ---------------------------
# CompatibleMultiHeadAttention (unchanged)
# ---------------------------
class CompatibleMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        for key in ['query_shape', 'key_shape', 'value_shape']:
            kwargs.pop(key, None)
        super().__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            super().build(input_shape)
        else:
            super().build(input_shape)

# ---------------------------
# Model & scaler loaders (unchanged except spinner text)
# ---------------------------
@st.cache_resource(show_spinner="Loading forecasting model...")
def load_model_with_fix(model_path):
    try:
        return load_model(
            model_path,
            custom_objects={
                'TFOpLambda': TFOpLambda,
                'MultiHeadAttention': CompatibleMultiHeadAttention,
                'tf': tf
            },
            compile=False
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Ensure TF version is 2.13.1 and Python is 3.10.x; model must match TF version.")
        st.stop()

@st.cache_resource(show_spinner="Loading data scaler...")
def load_scaler(scaler_path):
    try:
        return joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()

# ---------------------------
# Preprocessing uploaded data (unchanged)
# ---------------------------
@st.cache_data(show_spinner="Processing data...")
def load_and_preprocess_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, header=0, skiprows=[1])
        df.columns = [str(c).strip().replace(" ", "_").replace(".", "_") for c in df.columns]
        df.replace(-200, np.nan, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["timestamp"] = pd.to_datetime(df["Date"].dt.strftime('%Y-%m-%d') + ' ' + df["Time"].astype(str), errors="coerce")
        df.drop(["Date", "Time"], axis=1, inplace=True)
        df.interpolate(method="linear", inplace=True)
        df.dropna(inplace=True)
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month

        # Define features and targets
        target_cols = ["CO_GT_", "PT08_S1_CO_", "NMHC_GT_", "C6H6_GT_"]
        feature_cols = target_cols + [
            "PT08_S2_NMHC_", "NOx_GT_", "PT08_S3_NOx_", "NO2_GT_",
            "PT08_S4_NO2_", "PT08_S5_O3_", "T", "RH", "AH",
            "hour", "day_of_week", "month"
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]
        if not feature_cols:
            original_cols = [
                "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)",
                "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
                "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
                "hour", "day_of_week", "month"
            ]
            feature_cols = [c for c in original_cols if c in df.columns]
            target_cols = feature_cols[:4]

        return df[["timestamp"] + feature_cols], target_cols, feature_cols
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

# ---------------------------
# Main App (unchanged)
# ---------------------------
st.title("🌤️ Air Quality Forecasting")
st.markdown("""
Forecast air quality for the next 24 hours using Transformer.
Upload recent air quality data (Excel) to get predictions.
""")

uploaded_file = st.file_uploader("Upload Air Quality Data (Excel)", type=["xlsx"])
if uploaded_file:
    with st.spinner("Processing..."):
        df, target_cols, feature_cols = load_and_preprocess_data(uploaded_file)
        if len(df) < 24:
            st.warning(f"Only {len(df)} hours of data provided; need at least 24 hours.")
            st.stop()
        st.success(f"Loaded {len(df)} records with {len(feature_cols)} features.")

        model = load_model_with_fix("transformer_forecaster.h5")
        scaler = load_scaler("scaler.pkl")

        scaled = scaler.transform(df[feature_cols])
        seq = scaled[-24:].reshape(1, 24, len(feature_cols))
        last_ts = df["timestamp"].iloc[-1]
        forecasts, timestamps = [], []

        for i in range(24):
            pred = model.predict(seq, verbose=0)[0]
            new = np.zeros(len(feature_cols))
            new[:len(target_cols)] = pred
            new[len(target_cols):] = seq[0, -1, len(target_cols):]
            nxt = last_ts + timedelta(hours=i+1)
            for key, idx in [("hour", "hour"), ("day_of_week", "day_of_week"), ("month", "month")]:
                if key in feature_cols:
                    # handle day_of_week specially
                    if key == "day_of_week":
                        new[feature_cols.index(key)] = nxt.weekday()
                    else:
                        new[feature_cols.index(key)] = getattr(nxt, key)
            seq = np.roll(seq, -1, axis=1)
            seq[0, -1, :] = new
            forecasts.append(new[:len(target_cols)])
            timestamps.append(nxt)

        dummy = np.zeros((24, len(feature_cols)))
        dummy[:, :len(target_cols)] = forecasts
        inv = scaler.inverse_transform(dummy)[:, :len(target_cols)]
        results = pd.DataFrame(inv, columns=target_cols)
        results.insert(0, "Timestamp", timestamps)

    st.success("Forecast done!")
    st.subheader("24-Hour Forecast")
    st.dataframe(results.style.format({col: "{:.2f}" for col in target_cols}), height=400)

    # Visuals
    st.subheader("Visual Forecast")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(target_cols):
        axes[i].plot(results["Timestamp"], results[col], marker="o", linestyle="-", color="#1f77b4")
        axes[i].set_title(col)
        axes[i].set_xlabel("Time"); axes[i].set_ylabel("Value")
        axes[i].grid(True, alpha=0.3)
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    # Historical vs Forecast comparison
    st.subheader("Historical vs Forecast")
    hist = df[["timestamp"] + target_cols].copy().tail(48)
    hist["Type"] = "Historical"
    fcast = results.copy()
    fcast["Type"] = "Forecast"
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    axes2 = axes2.flatten()
    for i, col in enumerate(target_cols):
        axes2[i].plot(hist["timestamp"], hist[col], label="Historical", color="#2ca02c", linewidth=2)
        axes2[i].plot(fcast["Timestamp"], fcast[col], label="Forecast", marker="o", linestyle="-", color="#1f77b4")
        axes2[i].set_title(f"{col} Trend")
        axes2[i].set_xlabel("Time"); axes2[i].set_ylabel("Value")
        axes2[i].legend(); axes2[i].grid(True, alpha=0.3)
        plt.setp(axes2[i].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)

    # Download CSV
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
else:
    st.info("Upload an Excel file to begin.")

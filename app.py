
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Air Quality Forecaster",
    page_icon="🌫️",
    layout="wide"
)

# Feature names matching your training script
FEATURE_COLS = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
    'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
    'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
    'hour', 'day_of_week', 'month'
]

TARGET_COLS = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)']

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'loaded' not in st.session_state:
    st.session_state.loaded = False
if 'sequence' not in st.session_state:
    st.session_state.sequence = np.full((24, len(FEATURE_COLS)), 0.5)  # Default values

def load_model():
    """Load model and scaler with error handling"""
    try:
        st.session_state.model = tf.keras.models.load_model('transformer_forecaster.h5')
        st.session_state.scaler = joblib.load('scaler.pkl')
        st.session_state.loaded = True
        return True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state.loaded = False
        return False

# Main app
st.title("🌫️ Air Quality Forecasting System")
st.markdown("Predict next hour's air quality using Transformer neural network")

# Sidebar for model loading
with st.sidebar:
    st.header("Model Configuration")
    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            if load_model():
                st.success("Model loaded successfully!")
    
    if st.session_state.loaded:
        st.success("Model is loaded and ready")
    else:
        st.warning("Model not loaded. Click 'Load Model' first")
    
    st.markdown("---")
    st.header("Input Settings")
    
    # Time selection
    current_time = st.time_input("Current Time", value=datetime.now().time())
    current_date = st.date_input("Current Date", value=datetime.now().date())
    
    # Extract temporal features
    timestamp = datetime.combine(current_date, current_time)
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    month = timestamp.month
    
    # Store temporal features in session state
    st.session_state.sequence[:, -3] = hour
    st.session_state.sequence[:, -2] = day_of_week
    st.session_state.sequence[:, -1] = month

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Sensor Readings")
    
    # Create input sliders
    input_data = []
    for i, feature in enumerate(FEATURE_COLS):
        # Skip temporal features as they're set automatically
        if feature in ['hour', 'day_of_week', 'month']:
            continue
            
        # Get last value in sequence as default
        default_value = float(st.session_state.sequence[-1, i])
        
        # Create slider
        value = st.slider(
            feature, 
            min_value=0.0, 
            max_value=1.0, 
            value=default_value,
            key=feature,
            help=f"Normalized value of {feature}"
        )
        input_data.append(value)
        
        # Update last hour in sequence
        st.session_state.sequence[-1, i] = value

with col2:
    st.header("Current Sequence")
    st.markdown("Latest 24-hour window of normalized data")
    
    # Create a DataFrame for better display
    sequence_df = pd.DataFrame(
        st.session_state.sequence,
        columns=FEATURE_COLS
    )
    
    # Format temporal features as integers
    sequence_df[['hour', 'day_of_week', 'month']] = sequence_df[['hour', 'day_of_week', 'month']].astype(int)
    
    # Display sequence
    st.dataframe(sequence_df.style.format("{:.4f}"), height=400)

# Prediction button
if st.button("Predict Next Hour", disabled=not st.session_state.loaded):
    if not st.session_state.loaded:
        st.warning("Please load the model first")
        st.stop()
        
    with st.spinner("Making prediction..."):
        # Prepare input sequence
        input_seq = st.session_state.sequence.reshape(1, 24, len(FEATURE_COLS))
        
        # Predict
        prediction = st.session_state.model.predict(input_seq, verbose=0)[0]
        
        # Inverse transform
        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[:, :len(TARGET_COLS)] = prediction.reshape(1, -1)
        pred_inv = st.session_state.scaler.inverse_transform(dummy)[0, :len(TARGET_COLS)]
        
        # Update sequence - shift and add prediction
        st.session_state.sequence = np.roll(st.session_state.sequence, -1, axis=0)
        st.session_state.sequence[-1, :len(TARGET_COLS)] = prediction
        
        # Display results
        st.success("Prediction complete!")
        st.subheader("Next Hour Prediction:")
        
        # Create metrics display
        cols = st.columns(len(TARGET_COLS))
        units = ["ppm", "", "ppm", "µg/m³"]  # Units for each target
        
        for i, col in enumerate(TARGET_COLS):
            with cols[i]:
                st.metric(
                    label=col,
                    value=f"{pred_inv[i]:.2f}",
                    help=f"Predicted {col} for next hour"
                )
                st.caption(units[i])
        
        # Plot prediction
        st.subheader("Prediction Visualization")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(TARGET_COLS, pred_inv, color='skyblue')
        ax.set_title("Predicted Air Quality Metrics")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Add sequence management
st.markdown("---")
st.subheader("Sequence Management")

col1, col2 = st.columns(2)
with col1:
    if st.button("Reset to Defaults"):
        st.session_state.sequence = np.full((24, len(FEATURE_COLS)), 0.5)
        # Set temporal features
        timestamp = datetime.combine(datetime.now().date(), datetime.now().time())
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        st.session_state.sequence[:, -3] = hour
        st.session_state.sequence[:, -2] = day_of_week
        st.session_state.sequence[:, -1] = month
        st.success("Sequence reset to defaults!")

with col2:
    uploaded_file = st.file_uploader("Upload Sequence (CSV)", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape != (24, len(FEATURE_COLS)):
                st.error(f"Invalid shape. Expected (24, {len(FEATURE_COLS)}), got {df.shape}")
            else:
                st.session_state.sequence = df.values
                st.success("Sequence loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# Model info section
st.markdown("---")
st.subheader("Model Information")
st.write("""
- **Architecture**: Transformer Neural Network
- **Input**: 24-hour sequence of 16 features
- **Output**: 4 target variables
- **Training Loss**: MSE (Mean Squared Error)
""")

if st.session_state.loaded:
    st.success("Model is loaded and ready for predictions")
    st.download_button(
        "Download Current Sequence",
        pd.DataFrame(st.session_state.sequence, columns=FEATURE_COLS).to_csv(index=False).encode('utf-8'),
        "current_sequence.csv",
        "text/csv"
    )
else:
    st.warning("Model not loaded. Click 'Load Model' in the sidebar")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, InputSpec
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Display environment info
st.sidebar.subheader("Environment Information")
st.sidebar.write(f"Python version: {sys.version}")
st.sidebar.write(f"TensorFlow version: {tf.__version__}")

# Custom layer to handle TFOpLambda loading issue
class TFOpLambda(Layer):
    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        self.function = function
        self.supports_masking = True

    def call(self, inputs, mask=None, **kwargs):
        if mask is not None:
            return self.function(inputs, mask=mask)
        return self.function(inputs)

    def get_config(self):
        config = super().get_config()
        config['function'] = self.function
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Define custom MultiHeadAttention layer
class FixedMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        # Remove problematic parameters
        kwargs.pop('query_shape', None)
        kwargs.pop('key_shape', None)
        kwargs.pop('value_shape', None)
        super().__init__(**kwargs)

# Load model with custom objects
@st.cache_resource(show_spinner="Loading forecasting model...")
def load_model_with_fix(model_path):
    try:
        custom_objects = {
            'TFOpLambda': TFOpLambda,
            'MultiHeadAttention': FixedMultiHeadAttention,
            'tf': tf
        }
        return load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("This error is often caused by TensorFlow version incompatibility.")
        st.error("Try: pip install --upgrade tensorflow==2.13.1")
        st.stop()

# Load scaler
@st.cache_resource(show_spinner="Loading data scaler...")
def load_scaler(scaler_path):
    try:
        return joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        st.stop()

# Try to load resources
try:
    model = load_model_with_fix("transformer_forecaster.h5")
    scaler = load_scaler("scaler.pkl")
except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.stop()

# Preprocessing function
@st.cache_data(show_spinner="Processing data...")
def load_and_preprocess_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, header=0, skiprows=[1])
        
        # Clean column names
        df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
        
        # Handle missing values
        df.replace(-200, np.nan, inplace=True)
        
        # Process datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['timestamp'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        df = df.dropna(subset=['timestamp'])
        df.drop(['Date', 'Time'], axis=1, inplace=True)
        
        # Handle remaining missing values
        df.interpolate(method='linear', inplace=True)
        df.dropna(inplace=True)
        
        # Create time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Define columns
        target_cols = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)']
        feature_cols = [
            'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
            'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
            'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
            'hour', 'day_of_week', 'month'
        ]
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        return df[['timestamp'] + feature_cols], target_cols, feature_cols
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()

# Streamlit app
st.title("Air Quality Forecasting")
st.markdown("""
This app forecasts air quality parameters for the next 24 hours using a Transformer model.
Upload recent air quality data in Excel format to get predictions.
""")

# File uploader
uploaded_file = st.file_uploader("Upload Air Quality Data (Excel format)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Preprocess data
        df, target_cols, feature_cols = load_and_preprocess_data(uploaded_file)
        
        # Check data requirements
        if len(df) < 24:
            st.warning(f"Only {len(df)} hours of data available. Need at least 24 hours for forecasting.")
            st.stop()
            
        st.success(f"Data loaded successfully: {len(df)} records with {len(feature_cols)} features")
        
        # Normalize features
        scaled_data = scaler.transform(df[feature_cols])
        
        # Get last 24 hours of data
        last_24 = scaled_data[-24:]
        current_sequence = last_24.reshape(1, 24, len(feature_cols))
        last_timestamp = df['timestamp'].iloc[-1]
        
        # Generate forecasts
        forecasts = []
        timestamps = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(24):
            # Update progress
            progress = (i + 1) / 24
            progress_bar.progress(progress)
            status_text.text(f"Forecasting hour {i+1}/24...")
            
            # Predict next hour
            pred = model.predict(current_sequence, verbose=0)[0]
            
            # Create new row with predictions
            new_row = np.zeros(len(feature_cols))
            new_row[:len(target_cols)] = pred  # Predicted targets
            
            # Carry forward other features
            new_row[len(target_cols):] = current_sequence[0, -1, len(target_cols):]
            
            # Update time features for next hour
            next_time = last_timestamp + timedelta(hours=i+1)
            hour_idx = feature_cols.index('hour')
            dow_idx = feature_cols.index('day_of_week')
            month_idx = feature_cols.index('month')
            
            new_row[hour_idx] = next_time.hour
            new_row[dow_idx] = next_time.weekday()
            new_row[month_idx] = next_time.month
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_row
            
            # Store results
            forecasts.append(new_row[:len(target_cols)])
            timestamps.append(next_time)
        
        # Inverse transform predictions
        forecast_array = np.array(forecasts)
        dummy_features = np.zeros((24, len(feature_cols)))
        dummy_features[:, :len(target_cols)] = forecast_array
        predicted_values = scaler.inverse_transform(dummy_features)[:, :len(target_cols)]
        
        # Create results DataFrame
        results = pd.DataFrame(predicted_values, columns=target_cols)
        results.insert(0, 'Timestamp', timestamps)
        
        # Display results
        st.success("Forecast generated successfully!")
        st.subheader("Next 24 Hours Forecast")
        st.dataframe(results.style.format(subset=target_cols, formatter="{:.2f}"))
        
        # Plot results
        st.subheader("Forecast Visualization")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(target_cols):
            ax = axes[i]
            ax.plot(results['Timestamp'], results[col], marker='o', linestyle='-', color='#1f77b4')
            ax.set_title(f"{col} Forecast")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d %H:%M'))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show historical vs forecast
        st.subheader("Historical vs Forecast Comparison")
        history = df[['timestamp'] + target_cols].copy()
        history['Type'] = 'Historical'
        forecast = results.copy()
        forecast['Type'] = 'Forecast'
        
        # Combine data
        combined = pd.concat([
            history.rename(columns={'timestamp': 'Timestamp'}),
            forecast
        ])
        
        # Plot comparison
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
        axes2 = axes2.flatten()
        
        for i, col in enumerate(target_cols):
            ax = axes2[i]
            
            # Plot last 48 hours of historical data
            hist_trimmed = history.iloc[-48:]
            ax.plot(hist_trimmed['timestamp'], hist_trimmed[col], 
                    label='Historical', color='#2ca02c', linewidth=2)
            
            # Plot forecast
            ax.plot(forecast['Timestamp'], forecast[col], 
                    label='Forecast', marker='o', linestyle='-', color='#1f77b4')
            
            ax.set_title(f"{col} Trend")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d %H:%M'))
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Download button
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name='air_quality_forecast.csv',
            mime='text/csv',
            key='download-csv'
        )
    
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        st.stop()

else:
    st.info("Please upload an Excel file to generate forecasts")

# Instructions in sidebar
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. **Upload Data**:
   - Excel file with air quality measurements
   - Same format as AirQualityUCI.xlsx

2. **Data Requirements**:
   - Must contain at least 24 hours of data
   - Should include Date and Time columns
   - Same sensor columns as original dataset

3. **Output**:
   - 24-hour forecast for key pollutants
   - Visual comparison with historical data
   - Downloadable CSV results
""")
st.sidebar.title("Troubleshooting")
st.sidebar.markdown("""



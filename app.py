import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load your pre-trained model and scaler
model = load_model("transformer_forecaster.h5")
scaler = joblib.load("scaler.pkl")

# Define the same preprocessing function from your training script
def load_and_preprocess_data(uploaded_file):
    df = pd.read_excel(uploaded_file, header=0, skiprows=[1])
    
    df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
    
    df.replace(-200, np.nan, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['timestamp'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    df = df.dropna(subset=['timestamp'])
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    df.interpolate(method='linear', inplace=True)
    df.dropna(inplace=True)
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    target_cols = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)']
    feature_cols = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
        'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
        'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
        'hour', 'day_of_week', 'month'
    ]
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    return df[['timestamp'] + feature_cols], target_cols, feature_cols

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
        
        # Normalize features
        scaled_data = scaler.transform(df[feature_cols])
        
        # Check if enough data is available
        if len(df) < 24:
            st.warning(f"Only {len(df)} hours of data available. Need at least 24 hours for forecasting.")
        else:
            # Get last 24 hours of data
            last_24 = scaled_data[-24:]
            current_sequence = last_24.reshape(1, 24, len(feature_cols))
            last_timestamp = df['timestamp'].iloc[-1]
            
            # Generate forecasts
            forecasts = []
            timestamps = []
            
            for i in range(24):
                # Predict next hour
                pred = model.predict(current_sequence, verbose=0)[0]
                
                # Create new row with predictions
                new_row = np.zeros(len(feature_cols))
                new_row[:len(target_cols)] = pred  # Predicted targets
                
                # Carry forward other features
                new_row[len(target_cols):] = current_sequence[0, -1, len(target_cols):]
                
                # Update time features for next hour
                next_time = last_timestamp + timedelta(hours=i+1)
                new_row[feature_cols.index('hour')] = next_time.hour
                new_row[feature_cols.index('day_of_week')] = next_time.weekday()
                new_row[feature_cols.index('month')] = next_time.month
                
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
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(target_cols):
                ax = axes[i]
                ax.plot(results['Timestamp'], results[col], marker='o')
                ax.set_title(f"{col} Forecast")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.grid(True)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show historical vs forecast
            st.subheader("Historical Data vs Forecast")
            history = df[['timestamp'] + target_cols].copy()
            history['Type'] = 'Historical'
            forecast = results.copy()
            forecast['Type'] = 'Forecast'
            combined = pd.concat([history, forecast.rename(columns={'Timestamp': 'timestamp'})])
            
            fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
            axes2 = axes2.flatten()
            
            for i, col in enumerate(target_cols):
                ax = axes2[i]
                ax.plot(history['timestamp'], history[col], label='Historical', alpha=0.7)
                ax.plot(forecast['Timestamp'], forecast[col], label='Forecast', marker='o')
                ax.set_title(f"{col} Trend")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Forecast as CSV",
                data=csv,
                file_name='air_quality_forecast.csv',
                mime='text/csv'
            )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()

else:
    st.info("Please upload an Excel file to generate forecasts")

# Instructions in sidebar
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Upload an Excel file with air quality data
2. Ensure data format matches the original dataset:
   - Same columns as AirQualityUCI.xlsx
   - Contains Date and Time columns
3. The model requires at least 24 hours of historical data
4. Forecasts will be generated for the next 24 hours
""")
st.sidebar.markdown("Note: The model expects data with the same structure as the training data")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
import io

# Define the custom Transformer layer used in the model.
# This is required to load the saved model correctly.
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    
    # Feed-forward Network
    y = Dense(ff_dim, activation="relu")(x)
    y = Dense(inputs.shape[-1])(y)
    y = Dropout(dropout)(y)
    return LayerNormalization(epsilon=1e-6)(x + y)

# Define the data loading and preprocessing function from the original script
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """
    Loads and preprocesses the air quality data from an Excel file.
    It performs cleaning, interpolation, and feature engineering.
    """
    # Read the excel file, skipping the second row as per the original script
    df = pd.read_excel(uploaded_file, header=0, skiprows=[1])
    
    # Clean up column names
    df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
    
    # Replace -200 values with NaN
    df.replace(-200, np.nan, inplace=True)
    
    # Combine Date and Time columns into a single timestamp
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['timestamp'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # Interpolate missing values and drop remaining NaNs
    df.interpolate(method='linear', inplace=True)
    df.dropna(inplace=True)
    
    # Create time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Define target and feature columns
    target_cols = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)']
    feature_cols = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
        'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
        'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
        'hour', 'day_of_week', 'month'
    ]
    
    # Ensure all feature columns exist in the DataFrame
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    return df[['timestamp'] + feature_cols], target_cols, feature_cols

# Main Streamlit app logic
def main():
    st.title("Air Quality Forecasting App")
    st.write("This app uses a trained Transformer model to predict the next time step's air quality values based on historical data.")
    st.write("Please upload your `AirQualityUCI.xlsx` file to get started.")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a file", type="xlsx")

    if uploaded_file is not None:
        try:
            # Load and preprocess the data
            df, target_cols, feature_cols = load_and_preprocess_data(uploaded_file)
            st.success("File uploaded and data preprocessed successfully!")
            
            # Show a small preview of the data
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Load the pre-trained model and scaler
            # The custom_objects dictionary is crucial for loading the model with custom layers.
            try:
                model = tf.keras.models.load_model(
                    'transformer_forecaster.h5',
                    custom_objects={'transformer_encoder': transformer_encoder}
                )
                scaler = joblib.load('scaler.pkl')
                st.success("Model and scaler loaded successfully!")
            except FileNotFoundError:
                st.error("Error: Model files (`transformer_forecaster.h5` and `scaler.pkl`) not found. Please make sure they are in the same directory as this app.")
                return

            st.markdown("---")
            st.subheader("Prediction for the Next Time Step")

            # Check if there's enough data to create a sequence
            SEQ_LENGTH = 24
            if len(df) < SEQ_LENGTH:
                st.warning(f"Not enough data to make a prediction. The model requires at least {SEQ_LENGTH} data points.")
                return

            # Get the latest sequence of data for prediction
            latest_sequence_df = df[feature_cols].tail(SEQ_LENGTH)
            latest_timestamp = df['timestamp'].iloc[-1]
            st.write(f"Using the last **{SEQ_LENGTH}** data points ending at `{latest_timestamp}` to make a prediction.")
            
            # Normalize the latest sequence
            scaled_latest_sequence = scaler.transform(latest_sequence_df)
            
            # Reshape for the model (add a batch dimension)
            input_sequence = scaled_latest_sequence.reshape(1, SEQ_LENGTH, len(feature_cols))

            # Make the prediction
            with st.spinner('Predicting...'):
                normalized_prediction = model.predict(input_sequence)

            # Invert the normalization to get actual predicted values
            # Create a dummy array with the same shape as the scaler's training data
            dummy_array = np.zeros(shape=(1, len(feature_cols)))
            
            # Place the normalized prediction values into the dummy array
            for i in range(len(target_cols)):
                # Find the index of the target column in the full feature list
                try:
                    target_index = feature_cols.index(target_cols[i])
                    dummy_array[:, target_index] = normalized_prediction[0, i]
                except ValueError:
                    st.error(f"Error: Target column '{target_cols[i]}' not found in feature list.")
                    return

            # Inverse transform the dummy array
            inverted_prediction = scaler.inverse_transform(dummy_array)
            
            # Extract the relevant predicted values
            prediction_values = {}
            for i, col in enumerate(target_cols):
                target_index = feature_cols.index(col)
                prediction_values[col] = inverted_prediction[0, target_index]

            st.markdown("---")
            st.subheader("Predicted Air Quality Values")
            
            # Display the predictions in a user-friendly format
            st.markdown(f"The model predicts the following values for the time step after `{latest_timestamp}`:")
            for col, val in prediction_values.items():
                st.info(f"**{col}**: {val:.4f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()


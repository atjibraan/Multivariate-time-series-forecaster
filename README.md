# Multivariate-time-series-forecaster
# 🌫️ Air Quality Prediction App

This project is a **machine learning web app** built with **TensorFlow/Keras** and **Gradio** that predicts multiple air pollutant concentrations from 24 hours of environmental sensor readings.  
The model is based on the [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality).
Access the web app through [https://huggingface.co/spaces/AJibraan/MVt](url)

---

## ✨ Features
- 📂 **CSV Upload**: Upload a `.csv` file with 24 hours of sensor readings.
- 📋 **Paste Data Option**: Paste 24 rows of comma-separated values directly.
- 🔄 **Automatic Unit Conversion**: Outputs are converted back to their real-world units.
- 🧹 **Error Handling**: Handles placeholder values like `-200` from faulty sensors.
- 📊 **Multi-Pollutant Predictions**:
  - CO(GT)  
  - PT08.S1(CO)  
  - NMHC(GT)  
  - C6H6(GT)  

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/air-quality-prediction.git
cd air-quality-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

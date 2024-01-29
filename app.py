import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to load and preprocess the CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to create forecasts using Facebook Prophet
def create_forecast(data):
    # Rename columns for modeling
    data_for_modeling = data.rename(columns={data.columns[0]: "ds", data.columns[1]: "y"})

    # Fit the model
    model = Prophet()
    model.fit(data_for_modeling)
    
    # Create future dataframe for forecasting
    future = model.make_future_dataframe(periods=365)  # Forecast for the next year
    
    # Generate forecast
    forecast = model.predict(future)
    return forecast, model

def main():
    st.title("Forecasting with Prophet")

    # Create two columns for layout
    sample_col, upload_col = st.columns(2)

    # File preview for a local file
    sample_file_path = "pivot_ex.csv"  # Replace with your actual file path
    sample_col.subheader("Sample File:")
    sample_data = load_data(sample_file_path)
    sample_col.write(sample_data.head())

    # File upload
    uploaded_file = upload_col.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        # Center the preview and forecast components
        st.markdown("<h2>Preview of the Data</h2>", unsafe_allow_html=True)
        st.write(data.head(), style={'text-align': 'center'})
        
        forecast, model = create_forecast(data)

        # Plotting the forecast
        st.markdown("<h2>Forecast Components</h2>", unsafe_allow_html=True)
        fig_components = model.plot_components(forecast)
        st.write(fig_components)


# Run the app
if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load and preprocess data
df = pd.read_csv('copy_data.csv')
df['Day'] = pd.to_datetime(df['Day'])

df['Year'] = df['Day'].dt.year
df['Month'] = df['Day'].dt.month
df['Day'] = df['Day'].dt.day

features = ['Day', 'Month', 'Year', 'Avg_Humidity', 'Avg_Rainfall', 'Avg_Temperature']
target = 'Avg_Electric_Consumption'

X = df[features].values
y = df[target].values

# Normalize features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Load pre-trained model
model = load_model('final_model.h5')

# Prediction function
def predict_electric_consumption_range(start_date_str, end_date_str, humidity, rainfall, temp):
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    date_list = [start_date + timedelta(days=x) for x in range(0, (end_date - start_date).days + 1)]

    data_list = []
    for date_obj in date_list:
        day = date_obj.day
        month = date_obj.month
        year = date_obj.year
        data_list.append([day, month, year, humidity, rainfall, temp])

    new_data = pd.DataFrame(data_list, columns=['Day', 'Month', 'Year', 'Avg_Humidity', 'Avg_Rainfall', 'Avg_Temperature'])

    scaler_X.fit(X)
    X_new_scaled = scaler_X.transform(new_data)

    X_new_reshaped = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

    y_new_pred_scaled = model.predict(X_new_reshaped)
    
    scaler_y.fit(y.reshape(-1, 1))
    y_new_pred = scaler_y.inverse_transform(y_new_pred_scaled)

    prediction_df = pd.DataFrame({
        'Date': date_list,
        'Predicted_Electric_Consumption': y_new_pred.flatten()
    })
    
    return prediction_df, prediction_df['Predicted_Electric_Consumption'].sum()

# Function to plot interactive graph with Plotly
def plot_graph_with_plotly(predictions):
    fig = go.Figure()

    # Add predicted consumption line
    fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['Predicted_Electric_Consumption'], 
                             mode='lines+markers', 
                             name='Predicted Consumption',
                             line=dict(color='blue', width=2),
                             marker=dict(size=6, symbol='circle')))

    # Customize layout for better aesthetics
    fig.update_layout(
        title='Electric Consumption Prediction',
        xaxis_title='Date',
        yaxis_title='Predicted Electric Consumption',
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(tickformat='%d-%b-%Y', tickangle=45),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='gray'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')

    st.plotly_chart(fig, use_container_width=True)

# Streamlit app
st.set_page_config(page_title="Electric Power Consumption Predictor", layout="centered", initial_sidebar_state="auto")

# Adding custom CSS for enhanced UI
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 10px;
            font-family: 'Arial', sans-serif;
        }
        .css-1v0mbdj.e1tzin5v0 {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            width: 100%;
            font-size: 16px;
            padding: 10px 20px;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .stDateInput > div {
            background-color: #f0f2f6;
            padding: 10px;
        }
        .css-18e3th9 {
            background-color: #007bff;
            color: white;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .css-1ki0w8v {
            background-color: #007bff;
            color: white;
            padding: 10px;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.title("Configure Input Parameters")

# Date inputs in the sidebar
start_date = st.sidebar.date_input('Select the start date', datetime.today())
end_date = st.sidebar.date_input('Select the end date', datetime.today())

humidity = st.sidebar.slider('Average Humidity', 0.0, 100.0, 60.0)
rainfall = st.sidebar.slider('Average Rainfall', 0.0, 50.0, 10.0)
temperature = st.sidebar.slider('Average Temperature', -10.0, 50.0, 25.0)

start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Main title and description
st.title("🔌 Electric Power Consumption Predictor")
st.markdown("<h2 style='color: #007bff;'>Predict electric consumption based on historical data and visualize the predictions interactively.</h2>", unsafe_allow_html=True)

# Two columns for results and actions
col1, col2 = st.columns(2)

# Predict button
if col1.button('Predict'):
    output, total_required = predict_electric_consumption_range(start_date_str, end_date_str, humidity, rainfall, temperature)
    st.write(f"### Total predicted electric consumption: **{total_required:.2f} kWh**")
    st.dataframe(output)

# Graphical View button
if col2.button('Graphical View'):
    output, total_required = predict_electric_consumption_range(start_date_str, end_date_str, humidity, rainfall, temperature)
    plot_graph_with_plotly(output)

# Footer or credits
st.markdown("""
    <style>
        .footer {
            text-align: center;
            padding: 20px;
            background-color: #f0f2f6;
            color: #007bff;
            border-top: 1px solid #e1e1e1;
        }
    </style>
    <div class="footer">
        <p>Created with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

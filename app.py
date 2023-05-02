import streamlit.components.v1 as components
from keras.models import load_model
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import plotly.express as px
import requests
from datetime import datetime
from streamlit_lottie import st_lottie
from yfinance import shared

api_key = 'ef16897ad352427db59afd35cd4524aa'
ticker = 'MSFT' # replace with the ticker you want to download data for
start_date = '2010-01-01'
end_date = datetime.now()

url = f'https://api.twelvedata.com/time_series?symbol={ticker}&interval=1day&start_date={start_date}&end_date={end_date}&apikey={api_key}'

response = requests.get(url)

data = response.json()

df = pd.DataFrame.from_dict(data['values'])

df['datetime'] = pd.to_datetime(df['datetime'])

df.set_index('datetime', inplace=True)


fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df.close, mode='lines'))
fig.update_layout(title='Closing Price vs Time Chart',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    width=1000,
                    height=600)
st.plotly_chart(fig, use_container_width=True)

data_training = pd.DataFrame(df['close'][0:int(len(df)*70)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

    # load model
model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predict = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predict = y_predict * scale_factor
y_test = y_test * scale_factor

st.subheader('Original VS predicted')
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index[int(len(df)*0.70):], y=y_test, name='Original Price'))
fig2.add_trace(go.Scatter(x=df.index[int(len(df)*0.70):], y=y_predict[:, 0], name='Predict'))
fig2.update_layout(xaxis_title='Date',yaxis_title='Price',width=1000,height=600)
st.plotly_chart(fig2, use_container_width=True)
last_100_days = data_testing[-100:].values

    # Instantiate a scaler object and fit_transform the data
scaler = MinMaxScaler()
last_100_days_scaled = scaler.fit_transform(last_100_days)

    # Create an empty list to store the predicted prices
predicted_prices = []

    # Make predictions for the next day using the last 100 days of data
for i in range(1):
    X_test = np.array([last_100_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_price = model.predict(X_test)
    predicted_prices.append(predicted_price)
    last_100_days_scaled = np.append(last_100_days_scaled, predicted_price)
    last_100_days_scaled = np.delete(last_100_days_scaled, 0)

    # Invert the scaling of the predicted price
predicted_prices = np.array(predicted_prices)
predicted_prices = predicted_prices.reshape(predicted_prices.shape[0], predicted_prices.shape[2])
predicted_prices = scaler.inverse_transform(predicted_prices)

st.header('- Prediction')
st.write('Predicted price for the next day:', predicted_prices[0][0])
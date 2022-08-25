import streamlit as st
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import yfinance as yf
import matplotlib.pyplot as plt 

st.title('Learning Data Science — Predict Stock Price with Support Vector Regression (SVR)')
start = '2019-01-01'
end = '2019-12-31'
st.subheader('Preparación de la data')

df = yf.download('MSFT', start, end)
df = df.reset_index()
st.write(df)

df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

st.write(df)

plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Precio de cierre MSFT')
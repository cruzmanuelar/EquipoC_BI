import streamlit as st
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

st.write('Prueba')

bitcoin = pd.read_csv("bitcoin_price.csv",parse_dates=['Date'])
bitcoin = bitcoin.set_index('Date')
bitcoin.sort_index(inplace=True)
bitcoin.head()
import time
import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error
from pmdarima.arima import auto_arima
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

def app():

    st.title('Modelo - Arima')

    plt.style.use('fivethirtyeight')
    
    rcParams['figure.figsize'] = 20,10

    
    scaler = MinMaxScaler(feature_range=(0, 1))

    ticker='MSFT'
    period1 = int(time.mktime(datetime.datetime(2015, 1, 1, 0, 0).timetuple()))
    period2 = int(time.mktime(datetime.datetime.now().timetuple()))
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df_dis = pd.read_csv(query_string)

    data = df_dis.sort_index(ascending=True, axis=0)

    train = data[:1551]
    valid = data[1551:]

    training = train['Close']
    validation = valid['Close']

    model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
    model.fit(training)

    forecast = model.predict(n_periods=351)
    forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])
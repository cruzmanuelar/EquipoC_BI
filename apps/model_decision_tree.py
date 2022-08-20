import time
import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import streamlit as st

def app():
    plt.style.use('fivethirtyeight')

    #Establecemos el tamaño de la figura
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 20,10

    #Para la normalización de data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    ticker='MSFT'
    #Establecemos el año 2015
    period1 = int(time.mktime(datetime.datetime(2015, 1, 1, 0, 0).timetuple()))
    period2 = int(time.mktime(datetime.datetime.now().timetuple()))
    interval = '1d' # 1d, 1m
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)

    df['symbol']='MSFT'
    df.index=pd.to_datetime(df['Date'])
    df=df.drop(['Date'],axis='columns')
    st.df

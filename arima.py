import streamlit as st
import pandas as pd
# from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime


st.title('Modelo ARIMA')

ticker='AMZN'
period1 = int(time.mktime(datetime.datetime(2015, 1, 1, 0, 0).timetuple()))
period2 = int(time.mktime(datetime.datetime.now().timetuple()))
interval = '1d' # 1d, 1m
query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
df_amzn = pd.read_csv(query_string)

df_amzn['Date'] = pd.to_datetime(df_amzn.Date, format='%Y-%m-%d')
df_amzn.index = df_amzn['Date']

fig = plt.figure(figsize=(14,8))
plt.plot(df_amzn['Close'], label='Precio de cierre AMZN')

st.subheader('Visualización precio de cierre de amazon')
st.pyplot(fig)


data = df_amzn.sort_index(ascending=True, axis = 0)
st.write(data)

st.subheader('Últimos registros')

st.write(data.iloc[1920:1927])

st.subheader('Primeros registros de nuestra variable objetivo')
st.write(data['Close'].head())

train = data[:1343]
valid = data[1343:]

mod = ARIMA(data['Close'], order=(1,1,1))
res = mod.fit()
fig = res.plot_predict(start=train.shape[0],end=(train.shape[0]+valid.shape[0]+30), dynamic=False)
fig.set_size_inches(15, 8)

st.pyplot(fig)

# df_msft = yf.download('MSFT', start, end)
# df_msft = df_msft.reset_index()
# st.write(df_msft)
# st.write(df_msft.shape)

# st.pyplot(fig)
# st.subheader('Preparación de la data')

# df_msft['Date'] = pd.to_datetime(df_msft.Date, format='%Y-%m-%d')
# df_msft.index = df_msft['Date']

# st.write(df_msft)

# fig = plt.figure(figsize=(16,8))
# plt.plot(df_msft['Close'], label='Precio de cierre MSFT')
# # plt.show()
# st.pyplot(fig)

# data = df_msft.sort_index(ascending=True, axis = 0)
# st.write(data)

# nueva_data = pd.DataFrame(index=range(0, len(df_msft)), columns=['Date','Close'])

# for i in range(0, len(data)):
#   nueva_data['Date'][i] = data['Date'][i]
#   nueva_data['Close'][i] = data['Close'][i]

# st.write(nueva_data.head())

# train = nueva_data[:1348]
# valid = nueva_data[1348:]
# st.write(nueva_data.shape)
# st.write(train.shape)
# st.write(valid.shape)

# data = df_msft.sort_index(ascending=True, axis=0)

# train = data[:1348]
# valid = data[1341:]

# training = train['Close']
# validation = valid['Close']

# model = auto_arima(training, start_p=1, start_q=1,max_p=3, suppress_warning=True, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1, trace= True, error_action='ignore')
# model.fit(training)

# forecast = model.predict(n_periods=576)
# forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

# rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-np.array(forecast['Prediction'])),2)))
# st.write(rms)

# st.subheader('Preparación de la data')

# #plot
# plt.plot(train['Close'])
# plt.plot(valid['Close'])
# plt.plot(forecast['Prediction'])

# fig1 = plt.figure(figsize=(16,8))
# plt.plot(train['Close'], label='Precio de cierre MSFT')
# # plt.show()
# st.pyplot(fig1)

# fig2 = plt.figure(figsize=(16,8))
# plt.plot(valid['Close'], label='Precio de cierre MSFT')
# # plt.show()
# st.pyplot(fig2)

# fig3 = plt.figure(figsize=(16,8))
# plt.plot(forecast['Close'], label='Precio de cierre MSFT')
# # plt.show()
# st.pyplot(fig3)
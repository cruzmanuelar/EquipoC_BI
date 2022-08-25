import streamlit as st
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import yfinance as yf
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

st.title('Learning Data Science — Predict Stock Price with Support Vector Regression (SVR)')
start = '2015-01-02'
end = '2022-08-24'
st.subheader('Preparación de la data')

df_msft = yf.download('MSFT', start, end)
df_msft = df_msft.reset_index()
st.write(df_msft)
st.write(df_msft.shape)

df_msft['Date'] = pd.to_datetime(df_msft.Date, format='%Y-%m-%d')
df_msft.index = df_msft['Date']

st.write(df_msft)

fig = plt.figure(figsize=(16,8))
plt.plot(df_msft['Close'], label='Precio de cierre MSFT')
# plt.show()
st.pyplot(fig)

data = df_msft.sort_index(ascending=True, axis = 0)
st.write(data)

nueva_data = pd.DataFrame(index=range(0, len(df_msft)), columns=['Date','Close'])

for i in range(0, len(data)):
  nueva_data['Date'][i] = data['Date'][i]
  nueva_data['Close'][i] = data['Close'][i]

st.write(nueva_data.head())

train = nueva_data[:1348]
valid = nueva_data[1348:]
st.write(nueva_data.shape)
st.write(train.shape)
st.write(valid.shape)

data = df_msft.sort_index(ascending=True, axis=0)

train = data[:1348]
valid = data[1341:]

training = train['Close']
validation = valid['Close']

model = auto_arima(training, start_p=1, start_q=1,max_p=3, suppress_warning=True, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1, trace= True, error_action='ignore')
model.fit(training)

forecast = model.predict(n_periods=576)
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])
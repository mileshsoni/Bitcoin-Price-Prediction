import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models
from datetime import datetime, timedelta
import tensorflow as tf
import altair as alt
data = pd.read_csv('BTC_USD.csv')
data.drop(columns = ['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis = 1, inplace = True)
data['Date'] = [datetime.strptime(date, '%Y-%m-%d') for date in data['Date']]
graph_data = data.set_index('Date')
data.drop(['Date'], axis = 1, inplace = True)

st.header('Bitcoin Price Prediction')
st.subheader('Bitcoin Line Chart')
st.line_chart(graph_data)

lookup = 150
train_data = data.iloc[0:-lookup, :]
test_data = data.iloc[-lookup : , :]

scaler = MinMaxScaler(feature_range = (0,1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

model = models.load_model('model.keras')

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
test_data = pd.concat((train_data.tail(lookup), test_data), ignore_index = True)

def get_x_y ():
    x_test = []
    y_test = []
    for i in range(lookup, test_data.shape[0]):
        x_test.append(test_data.iloc[i-lookup : i, :])
        y_test.append(test_data.iloc[i, 0])
    return np.array(x_test), np.array(y_test)
x_test, y_test = get_x_y()

def get_next_date(last_date):
    current_date = datetime.strptime(last_date, '%Y-%m-%d')
    next_date = current_date + timedelta(days=1)
    next_date_str = next_date.strftime('%Y-%m-%d')
    return next_date_str

y_test = y_test.reshape(-1, 1)
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
y_pred_df = pd.DataFrame(y_pred)
y_test_df = pd.DataFrame(y_test)
table = pd.concat([y_pred_df, y_test_df], axis = 1)
table.columns = ['Predicted', 'Actual']

st.header('Predicted vs Actual Price')
st.markdown("### Chart")
st.line_chart(table)
st.markdown("### Values")
st.write(table)



last_date = graph_data.index[-1]
last_date = str(last_date).split()[0]

prices = test_data.iloc[-lookup :, :]
prices = np.array(prices)

prices = prices[:, 0]


dates = []
new_prices = []
future_days = 6
st.header(f'BTC Price Prediction for next {future_days} days')
for i in range(future_days):
    input_to_model = prices[-lookup:]
    input_to_model = np.reshape(input_to_model, (1, lookup, 1))
    y_pred = model.predict(input_to_model)
    new_prices.append(scaler.inverse_transform(y_pred))
    prices = np.append(prices, y_pred)
    dates.append(get_next_date(last_date))
    last_date = get_next_date(last_date)

dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
new_prices = np.reshape(new_prices, (-1, ))
st.markdown("### Chart")
df = pd.DataFrame({
    'Date': dates,
    'Price': new_prices
})
chart = alt.Chart(df).mark_line().encode(
    x='Date:T',
    y=alt.Y('Price:Q', scale=alt.Scale(domain=[df['Price'].min() - 5, df['Price'].max() + 5]))
).properties(
    width=700,
    height=400
)
# df.set_index('Date', inplace=True)
st.altair_chart(chart)
st.markdown("### Values")
df['Date'] = df['Date'].dt.date
df.set_index('Date', inplace = True)
st.write(df)

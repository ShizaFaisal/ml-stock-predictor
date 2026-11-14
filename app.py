import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("ML Stock Price Predictor")

ticker = st.text_input("Stock Symbol likho (e.g. AAPL, TSLA, NVDA)", "AAPL")

if st.button("Predict karo"):
    data = yf.download(ticker, period="1y", interval="1d")
    if len(data) == 0:
        st.error("Data nahi mila. Symbol check karo.")
    else:
        data = data.reset_index()
        data['Day'] = range(len(data))
        X = data[['Day']]
        y = data['Close']
        model = LinearRegression()
        model.fit(X, y)
        next_day = np.array([[len(data)]])
        pred = model.predict(next_day)
        st.success(f"Kal {ticker} ka predicted Close: ${pred[0]:.2f}")
        st.line_chart(data.set_index('Date')['Close'])

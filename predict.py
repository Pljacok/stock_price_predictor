import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

ticker = "AAPL"
data = yf.download(ticker, period="1y")
data["Prediction"] = data["Close"].shift(-1)

X = data[["Close"]].dropna()
y = data["Prediction"].dropna()

model = LinearRegression()
model.fit(X[:-1], y)

prediction = model.predict([[X.iloc[-1][0]]])
print(f"Прогноз ціни {ticker}: ${prediction[0]:.2f}")


# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 2. Load Dataset
data = pd.read_csv("data (1).csv")

# 3. Explore Dataset
print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())
print(data.describe())

# 4. Data Cleaning
data.columns = data.columns.str.strip()

if 'Company' in data.columns:
    data = data[data['Company'] == 'AAPL'].copy()

data['Date'] = pd.to_datetime(data['Date'], format='mixed')
for col in ['Close/Last', 'Open', 'High', 'Low']:
    data[col] = data[col].replace('[$,]', '', regex=True).astype(float)

data['Volume'] = pd.to_numeric(data['Volume'])
data = data.dropna().sort_values('Date')
data = data.set_index('Date')
data.rename(columns={'Close/Last': 'Close'}, inplace=True)

# 5. Historical Stock Price
data['Close'].plot(figsize=(10,5), title='Historical Stock Price')
plt.ylabel('Price ($)')
plt.grid()
plt.show()

# 6. Moving Averages
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

data[['Close', 'MA20', 'MA50']].plot(figsize=(10,5))
plt.title('Moving Averages')
plt.grid()
plt.show()

# 7. Seasonality Analysis
data.groupby(data.index.day_name())['Close'].mean().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday']
).plot(kind='bar', title='Average Price by Weekday')
plt.show()

data.groupby(data.index.month)['Close'].mean().plot(
    kind='bar', title='Monthly Average Price')
plt.show()

if len(data) >= 10:
    seasonal_decompose(data['Close'], model='additive', period=5).plot()
    plt.show()

# 8. Correlation Analysis
corr = data[['Close','Volume','Open','High','Low']].corr()
print(corr)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Stock Price Correlation')
plt.show()

plt.scatter(data['Volume'], data['Close'])
plt.xlabel('Volume')
plt.ylabel('Closing Price')
plt.title('Price vs Volume')
plt.show()

# 9. Forecasting
if len(data) >= 20:
    close = data['Close'].reset_index(drop=True)
    split = int(len(close) * 0.8)
    train, test = close[:split], close[split:]

    # ARIMA
    arima = ARIMA(train, order=(1,1,1)).fit()
    arima_forecast = arima.forecast(steps=len(test))

    # Exponential Smoothing
    exp = ExponentialSmoothing(train, trend='add').fit()
    exp_forecast = exp.forecast(len(test))

    # Forecast Graph
    plt.figure(figsize=(10,5))
    plt.plot(test.values, label='Actual')
    plt.plot(np.asarray(arima_forecast), label='ARIMA')
    plt.plot(np.asarray(exp_forecast), label='Exponential Smoothing')
    plt.title('Stock Price Forecast')
    plt.legend()
    plt.grid()
    plt.show()

    # Model Evaluation
    print("ARIMA RMSE:", np.sqrt(mean_squared_error(test, arima_forecast)))
    print("ARIMA MAE:", mean_absolute_error(test, arima_forecast))

    print("Exponential RMSE:", np.sqrt(mean_squared_error(test, exp_forecast)))
    print("Exponential MAE:", mean_absolute_error(test, exp_forecast))

    # Save Results
    pd.DataFrame({
        'Actual': test.values,
        'ARIMA': np.asarray(arima_forecast),
        'Exponential': np.asarray(exp_forecast)
    }).to_csv('Stock_Price_Forecasts.csv', index=False)

    print("Forecast results saved!")

else:
    print("Not enough records for forecasting.")

print("Stock Market Analysis Completed!")

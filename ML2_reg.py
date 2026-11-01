# 1. Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 2. Load dataset
df = pd.read_csv("uber2.csv.xls")

print("First 5 rows:")
print(df.head())

# 3. Remove unnecessary columns
df.drop(['Unnamed: 0', 'key'], axis=1, inplace=True)

# 4. Convert pickup_datetime into datetime format
df['pickup_datetime'] = pd.to_datetime(
    df['pickup_datetime'], errors='coerce', utc=True
)

# 5. Remove missing values
df.dropna(inplace=True)

# 6. Extract date and time features
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month

# 7. Validate latitude and longitude
df = df[
    df['pickup_latitude'].between(-90, 90) &
    df['dropoff_latitude'].between(-90, 90) &
    df['pickup_longitude'].between(-180, 180) &
    df['dropoff_longitude'].between(-180, 180)
]

# 8. Calculate distance using Haversine formula
lat1 = np.radians(df['pickup_latitude'])
lon1 = np.radians(df['pickup_longitude'])
lat2 = np.radians(df['dropoff_latitude'])
lon2 = np.radians(df['dropoff_longitude'])

a = (
    np.sin((lat2 - lat1) / 2) ** 2 +
    np.cos(lat1) * np.cos(lat2) *
    np.sin((lon2 - lon1) / 2) ** 2
)

df['distance'] = 2 * 6371 * np.arcsin(np.sqrt(a))

# 9. Remove unnecessary column
df.drop('pickup_datetime', axis=1, inplace=True)

# 10. Remove invalid records
df = df[
    (df['fare_amount'] > 0) &
    (df['passenger_count'] > 0) &
    (df['distance'] > 0) &
    (df['distance'] <= 130)
]

# 11. Detect outliers using boxplot
sns.boxplot(x=df['fare_amount'])
plt.title("Fare Amount Outlier Detection")
plt.show()

# 12. Remove fare outliers using IQR
Q1 = df['fare_amount'].quantile(0.25)
Q3 = df['fare_amount'].quantile(0.75)
IQR = Q3 - Q1

df = df[
    (df['fare_amount'] >= Q1 - 1.5 * IQR) &
    (df['fare_amount'] <= Q3 + 1.5 * IQR)
]

# 13. Display correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 14. Separate input and output variables
X = df.drop('fare_amount', axis=1)
y = df['fare_amount']

# 15. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=12
)

# 16. Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 17. Predict fare
y_pred = model.predict(X_test)

# 18. Evaluate model
print("R2 Score:", r2_score(y_test, y_pred))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# 19. Compare actual and predicted values
result = pd.DataFrame({
    'Actual Fare': y_test.values[:10],
    'Predicted Fare': y_pred[:10]
})

print("\nActual vs Predicted Fare:")
print(result)

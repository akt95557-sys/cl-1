
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 2. Load Dataset
data = pd.read_csv("C:/Users/prasa/Downloads/practical/aqi_data.csv")

# 3. Explore Dataset
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())
print(data.isnull().sum())

# 4. Clean Column Names
data.columns = (data.columns.str.strip().str.lower()
                .str.replace(r'[^a-z0-9]+', '_', regex=True)
                .str.strip('_'))

# Convert Date and Time
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Convert numeric columns
cols = ['pm2_5', 'pm10', 'no2', 'co', 'o3', 'so2', 'aqi_value']

for col in cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].fillna(data[col].median())

data.dropna(subset=['date'], inplace=True)
data = data.sort_values('date')

# 5. AQI Trend Over Time
daily_aqi = data.groupby('date')['aqi_value'].mean()

plt.figure(figsize=(9, 4))
plt.plot(daily_aqi.index, daily_aqi.values, marker='o', label='AQI')
plt.title("AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("AQI Value")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# 6. Individual Pollutant Trends
pollutants = ['pm2_5', 'pm10', 'co']

for col in pollutants:
    daily = data.groupby('date')[col].mean()
    plt.figure(figsize=(9, 4))
    plt.plot(daily.index, daily.values, marker='o', label=col.upper())
    plt.title(col.upper() + " Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Pollutant Level")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()

# 7. Bar Plot - Daily Average AQI
daily_aqi.plot(kind='bar', figsize=(9, 4), color='skyblue')
plt.title("Daily Average AQI")
plt.xlabel("Date")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Box Plot - AQI by Category
if 'aqi_category' in data.columns:
    categories = data['aqi_category'].dropna().unique()
    box_data = [
        data.loc[data['aqi_category'] == c, 'aqi_value'].dropna()
        for c in categories
    ]

    plt.figure(figsize=(8, 4))
    plt.boxplot(box_data, tick_labels=categories)
    plt.title("AQI Distribution by Category")
    plt.xlabel("AQI Category")
    plt.ylabel("AQI Value")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# 9. Scatter Plot - AQI vs Pollutants
for col in pollutants:
    plt.figure(figsize=(7, 4))
    plt.scatter(data[col], data['aqi_value'], alpha=0.6)
    plt.title("AQI vs " + col.upper())
    plt.xlabel(col.upper())
    plt.ylabel("AQI Value")
    plt.grid()
    plt.tight_layout()
    plt.show()

# 10. Save Cleaned Dataset
data.to_csv("Cleaned_City_Air_Quality.csv", index=False)

print("Air Quality Data Visualization Completed!")

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. API Key and Location
api_key = "fb365aa6104829b44455572365ff3b4e"
lat = 18.184135
lon = 74.610764

# 2. Retrieve Weather Data
url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"

response = requests.get(url)

if response.status_code != 200:
    print("API Error:", response.text)
    raise SystemExit

data = response.json()
print("Weather Data Retrieved Successfully")

# 3. Extract Weather Attributes
records = []

for x in data['list']:
    records.append([
        x['dt_txt'],
        x['main']['temp'],
        x['main']['humidity'],
        x['main']['pressure'],
        x['wind']['speed'],
        x.get('rain', {}).get('3h', 0)
    ])

df = pd.DataFrame(records, columns=[
    'DateTime', 'Temperature', 'Humidity',
    'Pressure', 'WindSpeed', 'Rainfall'
])

# 4. Data Cleaning
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nWeather Dataset:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# 5. Weather Data Analysis
print("\nWeather Statistics:")
print("Average Temperature:", df['Temperature'].mean())
print("Maximum Temperature:", df['Temperature'].max())
print("Minimum Temperature:", df['Temperature'].min())
print("Average Humidity:", df['Humidity'].mean())
print("Average Wind Speed:", df['WindSpeed'].mean())

# 6. Visualization - Temperature
plt.figure(figsize=(9, 4))
plt.plot(df['DateTime'], df['Temperature'], marker='o')
plt.title("Temperature Changes Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Humidity and Wind Speed
df.plot(x='DateTime', y=['Humidity', 'WindSpeed'],
        figsize=(9, 4))
plt.title("Humidity and Wind Speed")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Rainfall
plt.figure(figsize=(9, 4))
plt.bar(df['DateTime'].dt.strftime('%m-%d %H:%M'),
        df['Rainfall'])
plt.title("Rainfall Analysis")
plt.xlabel("Date and Time")
plt.ylabel("Rainfall (mm)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 7. Daily Aggregation
df['Date'] = df['DateTime'].dt.date

daily = df.groupby('Date').agg({
    'Temperature': 'mean',
    'Humidity': 'mean',
    'Rainfall': 'sum',
    'WindSpeed': 'mean'
})

print("\nDaily Weather Summary:")
print(daily)

# 8. Geographical Visualization
print("\nLocation:", data['city']['name'])
print("Latitude:", lat)
print("Longitude:", lon)

plt.figure(figsize=(6, 5))
plt.scatter(lon, lat, s=200, marker='o')
plt.text(lon, lat, " Weather Location")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographical Location of Weather Data")
plt.grid()
plt.show()

# 9. Correlation Analysis
plt.figure(figsize=(7, 5))
sns.heatmap(
    df[['Temperature', 'Humidity', 'Pressure',
        'WindSpeed', 'Rainfall']].corr(),
    annot=True, cmap='coolwarm', fmt='.2f'
)
plt.title("Weather Parameter Correlation")
plt.show()

print("\nWeather Analysis Completed Successfully!")
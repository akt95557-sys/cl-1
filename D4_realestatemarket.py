import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("bhp.csv") 

df.head()
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

df = df.dropna()

df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

def convert_sqft(x):
    try:
        values = x.split('-')
        if len(values) == 2:
            return (float(values[0]) + float(values[1])) / 2
        return float(x)
    except:
        return np.nan


df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna(subset=['total_sqft'])

# Calculate price per square feet
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
# Clean location
df['location'] = df['location'].str.strip()
counts = df['location'].value_counts()
df['location'] = df['location'].apply(
    lambda x: 'other' if counts[x] <= 10 else x
)

# Remove unrealistic data
df = df[df.total_sqft / df.bhk >= 300]
# Remove outliers using IQR
for col in ['total_sqft', 'bath', 'price', 'bhk', 'price_per_sqft']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) &
            (df[col] <= Q3 + 1.5 * IQR)]

print("Dataset Shape:", df.shape)
print("Price Count:", df['price'].count())
print(df['price'].describe())

df = pd.read_csv("bhp.csv")
plt.figure(figsize=(6, 4))
plt.boxplot(df['price'].dropna())
plt.title("Boxplot of House Prices")
plt.ylabel("Price")
plt.show()

for col in ['total_sqft', 'bath', 'price', 'bhk', 'price_per_sqft']:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col].dropna())
    plt.title("Boxplot of " + col)
    plt.show()

# Display cleaned dataset
print("\nCleaned Dataset:\n", df.head())
print("\nFinal Shape:", df.shape)
print("\nMissing Values:\n", df)

print("\nAverage Price by Location:")
print(df.groupby('location')['price'].mean().head(10))

df['bhk'].value_counts().sort_index().plot(kind='bar')
plt.title("Property Type Distribution")
plt.xlabel("BHK")
plt.ylabel("Number of Properties")
plt.show()

print("\nAverage Price by BHK:")
print(df.groupby('bhk')['price'].mean())

print("\nCleaned Dataset:")
print(df.head())
print("\nFinal Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
# 17. Save Cleaned Dataset
df.to_csv("Cleaned_Real_Estate.csv", index=False)
print("\nCleaned Dataset Saved Successfully!")



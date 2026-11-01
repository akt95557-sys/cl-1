import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import seaborn as sns
%matplotlib inline
sns.set(style="whitegrid")

# from google.colab import drive
# drive.mount('/content/drive')
# File 1: CSV
csv = pd.read_csv("sales_data_sample_d1.csv", encoding="cp1252")
print("File 1: 'sales_data_sample.csv' loaded.")


# File 2: XLSX
ed = pd.read_excel("Sample-Sales-Datad_d1.xlsx")
print("File 2: 'Sample-Sales-Data.xlsx' loaded.")
# File 3: JSON
with open("customers_d1.json", "r") as json_file:
    json_data = json.load(json_file)
print("File 3: 'customers.json' loaded.")

# Convert JSON data into a DataFrame
cust_df = pd.DataFrame(json_data)
print("JSON data has been converted into a DataFrame (cust_df).")
print("--- File 1 (csv) Info ---")
csv.info()
print("--- File 1 (csv) Describe ---")
csv.describe()
print("--- File 1 (csv) shape ---")
csv.shape
print("--- File 1 (csv) head (first few rows) ---")
csv.head()
print("--- File 1 (csv) tail (last few rows) ---")
csv.tail()
print("\n--- File 2 (ed) Info ---")
ed.info()
print("\n--- File 2 (ed) describe ---")
ed.describe()
print("\n--- File 2 (ed) shape ---")
ed.shape
print("--- File 2 (ed) head (first few rows) ---")
ed.head()
print("--- File 2 (ed) tail (last few rows) ---")
ed.tail()
print("\n--- File 3 json (cust_df) Info ---")
cust_df.info()
print("\n--- File 3 json (cust_df) describe ---")
cust_df.describe()
print("\n--- File 3 json (cust_df) shape ---")
cust_df.shape
print("--- File 3 json (cust_df) head (first few rows) ---")
cust_df.head()
print("--- File 3 json (cust_df) tail (last few rows) ---")
cust_df.tail()
print("--- Cleaning Main Sales Data (CSV file) ---")

# 1. Convert 'ORDERDATE' to datetime format (as observed in csv.info())
csv['ORDERDATE'] = pd.to_datetime(csv['ORDERDATE'])
print("'ORDERDATE' column has been converted to datetime format.")

# 2. Fill missing 'TERRITORY' values with 'Unknown' (as seen in csv.info())
csv['TERRITORY'] = csv['TERRITORY'].fillna('Unknown')
print("Missing 'TERRITORY' values have been filled with 'Unknown'.")

# 3. Create a 'YEAR_MONTH' column for time-series analysis
csv['YEAR_MONTH'] = csv['ORDERDATE'].dt.to_period('M')

print("\nCleaning complete. The 'csv' DataFrame is now ready for analysis.")
print("--- Calculating Key Metrics ---")

# 1. Total Revenue
total_revenue = csv['SALES'].sum()
print(f"Total Revenue: ${total_revenue:,.2f}")

# 2. Product-wise Sales
product_sales = csv.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False)
print("\n--- Product-wise Sales ---")
print(product_sales.to_string(float_format="${:,.2f}".format))

# 3. Customer Trends (Top 10)
customer_trends = csv.groupby('CUSTOMERNAME')['SALES'].sum().sort_values(ascending=False).head(10)
print("\n--- Top 10 Customers ---")
print(customer_trends.to_string(float_format="${:,.2f}".format))

# 4. Monthly Sales (for line chart visualization)
monthly_sales = csv.groupby('YEAR_MONTH')['SALES'].sum()
monthly_sales.index = monthly_sales.index.astype(str)  # Convert index to string for plotting

print("--- Visualization 1: Product-wise Sales ---")
plt.figure(figsize=(10, 6))
product_sales.plot(kind='bar', color=sns.color_palette('viridis'))
plt.title('Total Sales by Product Line', fontsize=16)
plt.ylabel('Total Sales (in millions)')
plt.xlabel('Product Line')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- Visualization 2: Monthly Sales Trends ---")
plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line', marker='o', linestyle='-')
plt.title('Monthly Sales Over Time', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Visualization 3: Sales by Territory ---")
# Calculate the metric for the pie chart
territory_sales = csv.groupby('TERRITORY')['SALES'].sum()
plt.figure(figsize=(5, 5))
territory_sales.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Sales Distribution by Territory', fontsize=16)
plt.ylabel('')  # Remove the default y-axis label
plt.axis('equal')  # Ensure the pie chart is drawn as a circle
plt.tight_layout()
plt.show()

print("\n--- Visualization 4: Sales Value Distribution ---")
csv['SALES'].plot(kind='hist', bins=50, figsize=(10, 5), color='blue', edgecolor='black')
plt.title('Sales Value Distribution')
plt.xlabel('Sales Value')
plt.ylabel('Frequency (Number of Occurrences)')
plt.show()
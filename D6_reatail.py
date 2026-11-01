
# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load Dataset
data = pd.read_csv("customer_shopping_data.csv")

# 3. Explore Dataset
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())
print(data.isnull().sum())

# 4. Clean Dataset
data.columns = data.columns.str.strip().str.lower()

data['price'] = pd.to_numeric(data['price'], errors='coerce')
data['quantity'] = pd.to_numeric(data['quantity'], errors='coerce')

data.dropna(subset=['shopping_mall', 'category',
                    'price', 'quantity'], inplace=True)

# Calculate Total Sales
data['total_sales'] = data['price'] * data['quantity']

# 5. Total Sales by Shopping Mall
mall_sales = data.groupby('shopping_mall')['total_sales'].sum()
mall_sales = mall_sales.sort_values(ascending=False)

print("\nTotal Sales by Shopping Mall:")
print(mall_sales)

# 6. Bar Chart - Sales by Mall
mall_sales.plot(kind='bar', color='skyblue', figsize=(9, 5))
plt.title("Sales Performance by Shopping Mall")
plt.xlabel("Shopping Mall")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie Chart - Sales Distribution
mall_sales.plot(kind='pie', autopct='%1.1f%%', figsize=(7, 7))
plt.title("Sales Distribution by Shopping Mall")
plt.ylabel("")
plt.show()

# 7. Top Performing Mall
print("\nTop Performing Mall:")
print(mall_sales.idxmax())

print("Highest Sales:")
print(mall_sales.max())

# 8. Sales by Mall and Product Category
category_sales = data.pivot_table(
    index='shopping_mall',
    columns='category',
    values='total_sales',
    aggfunc='sum',
    fill_value=0
)

print("\nSales by Mall and Category:")
print(category_sales)

# 9. Stacked Bar Chart
category_sales.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Sales by Shopping Mall and Product Category")
plt.xlabel("Shopping Mall")
plt.ylabel("Total Sales")
plt.legend(title="Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 10. Save Aggregated Data
mall_sales.to_csv("Mall_Sales_Summary.csv")
category_sales.to_csv("Mall_Category_Sales.csv")

print("\nData Aggregation Completed Successfully!")

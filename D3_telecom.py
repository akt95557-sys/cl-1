# 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# 2. Load Dataset
data = pd.read_csv("Telcom_Customer_Churn.csv")
print("Dataset Loaded Successfully")

# 3. Explore Dataset
print("\nDataset Information:")
data.info()

print("\nStatistical Description:")
print(data.describe())

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Columns:")
print(data.columns)

print("\nFirst 5 Rows:")
print(data.head())

print("\nLast 5 Rows:")
print(data.tail())


# 4. Check Missing Values and Remove Duplicates

print("\nMissing Values:")
print(data.isna().sum())

print("\nRows Before Removing Duplicates:", data.shape[0])
data.drop_duplicates(inplace=True)
print("Rows After Removing Duplicates:", data.shape[0])


# 5. Encode Categorical Data
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# 6. Standardize Numerical Data
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
data[num_cols] = scaler.fit_transform(data[num_cols])
print("\nData Types After Preprocessing:")
print(data.dtypes)

# 7. Correlation Heatmap
plt.figure(figsize=(19, 19))
sns.heatmap(
    data.corr(),
    annot=True,
    cmap='coolwarm',
    linewidths=0.6
)
plt.title("Correlation Heatmap of Features")
plt.show()

# 8. Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data)
plt.title("Distribution of Churned vs Non-Churned Customers")
plt.show()

# 9. Tenure vs Churn
plt.figure(figsize=(6, 4))
sns.boxplot(x='Churn', y='tenure', data=data)
plt.title("Tenure vs Churn")
plt.show()

# 10. Separate Features and Target
X = data.drop('Churn', axis=1)
y = data['Churn']

# 11. Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 12. Initialize Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# 13. Train and Evaluate Models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print("\n", name, "Accuracy:", round(acc, 4))
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

# 14. Model Accuracy Comparison
plt.figure(figsize=(6, 4))
sns.barplot(
    x=list(results.keys()),
    y=list(results.values())
)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()

# 15. Random Forest Feature Importance
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
sns.barplot(
    x=X.columns[indices],
    y=importances[indices]
)
plt.title("Feature Importance (Random Forest)")
plt.xticks(rotation=45, ha='right')
plt.show()

# 16. Confusion Matrix - Random Forest
y_pred_rf = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("\nCustomer Churn Analysis Completed Successfully!")
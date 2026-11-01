# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 2. Load dataset
data = pd.read_csv(
    "car_evaluation5.csv",
    names=['buying', 'maint', 'doors', 'persons',
           'lug_boot', 'safety', 'class']
)

print("First 5 rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())

# 3. Encode categorical data
encoders = {}

for col in data.columns:
    encoders[col] = LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])

# 4. Separate features and target
X = data.drop('class', axis=1)
y = data['class']

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0
)

# 6. Apply Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=0
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Prediction
y_pred = model.predict(X_test)

# 9. Calculate accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# 10. Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 11. Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# 12. Plot feature importance
sns.barplot(x=importance.values, y=importance.index)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Important Features for Car Evaluation")
plt.show()
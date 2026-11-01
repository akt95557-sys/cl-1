
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("Wine1.csv")

# Separate features and target
X = df.drop('Customer_Segment', axis=1)
y = df['Customer_Segment']

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=pd.factorize(y)[0], cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Wine Dataset')
plt.colorbar()
plt.show()

# Explained variance
print("Explained Variance:", pca.explained_variance_ratio_)
print("Total Variance:", sum(pca.explained_variance_ratio_))






code 2


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("Wine1.csv")

# Separate features and target
X = df.drop('Customer_Segment', axis=1)
y = df['Customer_Segment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=12
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Classification without PCA
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Without PCA:")
print(classification_report(y_test, y_pred))

# Apply PCA
pca = PCA(n_components=2)
tr_comp = pca.fit_transform(X_train)
ts_comp = pca.transform(X_test)

# Classification with PCA
pc_model = LogisticRegression()
pc_model.fit(tr_comp, y_train)
y_pred = pc_model.predict(ts_comp)

print("With PCA:")
print(classification_report(y_test, y_pred))

# PCA visualization
plt.scatter(tr_comp[:, 0], tr_comp[:, 1],
            c=pd.factorize(y_train)[0], cmap='viridis')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Principal Components')
plt.colorbar()
plt.show()

# Variance
print("Explained Variance:", pca.explained_variance_ratio_)
print("Principal Components:", pca.components_)

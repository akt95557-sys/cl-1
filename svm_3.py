
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load handwritten digits dataset
digits = datasets.load_digits()

X = digits.data
y = digits.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict digits
y_pred = model.predict(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Display first 10 test images
plt.figure(figsize=(10, 3))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title("Pred: " + str(y_pred[i]))
    plt.axis('off')

plt.suptitle("SVM Digit Classification")
plt.show()

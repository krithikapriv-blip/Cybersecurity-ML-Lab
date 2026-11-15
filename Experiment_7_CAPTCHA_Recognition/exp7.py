
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load handwritten digit dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Calculate accuracy
accuracy = model.score(X_test, y_test) * 100
print("Model Accuracy:", round(accuracy, 2), "%")

# Predict four digits
sample = X_test[:4]
prediction = model.predict(sample)

print("Predicted CAPTCHA:", "".join(map(str, prediction)))

# Display predicted digits
fig, ax = plt.subplots(1, 4, figsize=(6, 2))

for i in range(4):
    ax[i].imshow(sample[i].reshape(8, 8), cmap="gray")
    ax[i].axis("off")
    ax[i].set_title(prediction[i])

plt.suptitle("ML-Based CAPTCHA Recognition")
plt.tight_layout()

from pathlib import Path
plt.savefig(Path(__file__).with_name("captcha_output.png"), dpi=300)

plt.show()

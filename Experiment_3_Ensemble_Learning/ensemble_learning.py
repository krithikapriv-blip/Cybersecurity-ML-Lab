import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample cybersecurity dataset
data = {
    "packet_size": [100, 120, 150, 200, 500, 600, 700, 800, 110, 130, 550, 750],
    "packet_rate": [10, 12, 15, 20, 80, 90, 100, 110, 11, 14, 85, 105],
    "connection_count": [2, 3, 4, 5, 30, 35, 40, 45, 2, 3, 32, 42],
    "attack": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Select features
X = df[["packet_size", "packet_rate", "connection_count"]]

# Select target
y = df["attack"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

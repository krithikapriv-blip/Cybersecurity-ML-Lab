import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample URL dataset
data = {
    "url_length": [
        20, 25, 30, 35, 40, 45,
        80, 90, 100, 110, 120, 130,
        22, 28, 33, 38
    ],

    "dots": [
        1, 1, 2, 1, 2, 2,
        5, 6, 7, 8, 6, 9,
        1, 2, 1, 2
    ],

    "hyphens": [
        0, 0, 1, 0, 1, 0,
        4, 5, 6, 5, 7, 8,
        0, 1, 0, 1
    ],

    "digits": [
        0, 0, 1, 0, 2, 1,
        8, 10, 12, 15, 13, 18,
        0, 1, 0, 1
    ],

    "at_symbol": [
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0
    ],

    "ip_address": [
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0
    ],

    "label": [
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Select features
X = df.drop("label", axis=1)

# Select target
y = df["label"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

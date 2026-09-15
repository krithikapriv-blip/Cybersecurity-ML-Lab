import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

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

df = pd.DataFrame(data)

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="linear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("SVM Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

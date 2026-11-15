import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Sample database input dataset
data = {
    'special_characters': [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    'sql_keywords':       [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    'input_length':       [10, 15, 80, 90, 75, 12, 85, 20, 95, 18],
    'label':              [0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Test database inputs
predictions = model.predict(X)

print("Malicious Database Input Detection")
print("----------------------------------")

for i, result in enumerate(predictions):
    status = "Malicious Input" if result == 1 else "Safe Input"
    print("Input", i + 1, ":", status)

# Test a new input
new_input = [[1, 1, 85]]
result = model.predict(new_input)

print("\nNew Input:", 
      "Malicious Input" if result[0] == 1 else "Safe Input")

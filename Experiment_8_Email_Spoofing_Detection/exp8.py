import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Sample email dataset
data = {
    'domain_match': [1,1,1,0,0,1,0,1,0,1,1,0,1,0,1,0],
    'reply_match':  [1,1,1,0,0,1,0,1,0,1,1,0,1,0,1,0],
    'spf_pass':     [1,1,1,0,0,1,0,1,0,1,1,0,1,0,1,0],
    'dkim_pass':    [1,1,1,0,1,1,0,1,0,1,1,0,1,0,1,0],
    'suspicious':   [0,0,0,1,1,0,1,0,1,0,0,1,0,1,0,1],
    'label':        [0,0,0,1,1,0,1,0,1,0,0,1,0,1,0,1]
}

df = pd.DataFrame(data)

X = df.drop('label', axis=1)
y = df['label']

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Predict email status
prediction = model.predict(X)

for i, result in enumerate(prediction):
    status = "Spoofed" if result == 1 else "Genuine"
    print("Email", i+1, ":", status)

# Test a new email
new_email = [[0, 0, 0, 0, 1]]
result = model.predict(new_email)

print("\nNew Email:", "Spoofed" if result[0] == 1 else "Genuine")

with open("email_spoofing_output.txt", "w") as file:
    for i, prediction_result in enumerate(prediction):
        status = "Spoofed" if prediction_result == 1 else "Genuine"
        file.write(f"Email {i+1}: {status}\n")

    file.write("\nNew Email: " + 
               ("Spoofed" if result[0] == 1 else "Genuine"))

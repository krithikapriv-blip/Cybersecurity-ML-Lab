from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample email dataset
emails = [
    "Win a free prize now",
    "Claim your lottery reward",
    "Get free money today",
    "Congratulations you won a prize",
    "Meeting scheduled for tomorrow",
    "Please find the project report",
    "Let's discuss the assignment",
    "Your class starts at 10 AM"
]

labels = [
    "Spam", "Spam", "Spam", "Spam",
    "Not Spam", "Not Spam", "Not Spam", "Not Spam"
]

# Convert email text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

# Test a new email
new_email = ["Congratulations! You won free money"]
new_email_vector = vectorizer.transform(new_email)

prediction = model.predict(new_email_vector)

print("Email:", new_email[0])
print("Prediction:", prediction[0])

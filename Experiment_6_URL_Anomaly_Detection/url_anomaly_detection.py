import pandas as pd
import re
from sklearn.ensemble import IsolationForest

urls = [
    "https://google.com",
    "https://amazon.com",
    "https://wikipedia.org",
    "https://microsoft.com",
    "https://github.com",
    "https://apple.com",
    "https://example.com/login",
    "https://facebook.com",
    "https://youtube.com",
    "https://linkedin.com",
    "http://192.168.1.100/login/account/verify",
    "http://example.com/very-long-login-page-account-verification-password-reset",
    "http://192.168.10.50/user/login/123456789",
    "http://secure-login-example.com/account/verify",
    "http://example.com"
]


def extract_features(url):
    length = len(url)
    dots = url.count(".")
    hyphens = url.count("-")
    digits = sum(char.isdigit() for char in url)

    special_chars = len(
        re.findall(r"[^a-zA-Z0-9]", url)
    )

    at_symbol = url.count("@")

    ip_address = 1 if re.search(
        r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
        url
    ) else 0

    return [
        length,
        dots,
        hyphens,
        digits,
        special_chars,
        at_symbol,
        ip_address
    ]


features = []

for url in urls:
    features.append(extract_features(url))


df = pd.DataFrame(
    features,
    columns=[
        "length",
        "dots",
        "hyphens",
        "digits",
        "special_chars",
        "at_symbol",
        "ip_address"
    ]
)

print("Extracted Features:")
print(df)


model = IsolationForest(
    contamination=0.2,
    random_state=42
)

model.fit(df)

df["anomaly"] = model.predict(df)

df["result"] = df["anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})


print("\nURL Anomaly Detection Results:")

for i in range(len(urls)):
    print(
        urls[i],
        "->",
        df["result"].iloc[i]
    )

import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split



TRAIN_FILE = r"C:\Users\rishi\Downloads\NLP DATASET\processed_data\train_processed.json"

DEV_FILE = r"C:\Users\rishi\Downloads\NLP DATASET\processed_data\dev_processed.json"




def load_processed_json(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    claims = []
    labels = []

    for item in data:

        claims.append(item["claim"])
        labels.append(item["label"])

    return claims, labels




print("\nLoading training data...")

train_claims, train_labels = load_processed_json(TRAIN_FILE)

print("Training samples:", len(train_claims))




print("\nLoading dev data...")

dev_claims, dev_labels = load_processed_json(DEV_FILE)

print("Dev samples:", len(dev_claims))




print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train = vectorizer.fit_transform(train_claims)

X_dev = vectorizer.transform(dev_claims)




print("\nTraining Logistic Regression Model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, train_labels)




print("\nPredicting on dev set...")

predictions = model.predict(X_dev)




accuracy = accuracy_score(dev_labels, predictions)



print(f"\nAccuracy: {accuracy:.4f}")


print("\nClassification Report:\n")

print(classification_report(dev_labels, predictions))


print("\nConfusion Matrix:\n")

print(confusion_matrix(dev_labels, predictions))


label_map_reverse = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO"
}

while True:

    user_claim = input("\nEnter a claim (or type exit): ")

    if user_claim.lower() == "exit":
        break

    user_vector = vectorizer.transform([user_claim])

    pred = model.predict(user_vector)[0]

    print(f"\nPrediction: {label_map_reverse[pred]}")
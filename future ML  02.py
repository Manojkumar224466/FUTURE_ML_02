# ===============================
# 1. Import Libraries
# ===============================

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ===============================
# 2. Create Sample Dataset
# ===============================

data = {
    "ticket_text": [
        "My payment failed but money was deducted",
        "I cannot login to my account",
        "Website is very slow and not loading",
        "How to update my profile information",
        "Refund not received yet",
        "App crashes when I open it",
        "Unable to reset my password",
        "Delivery was delayed by two weeks",
        "Need invoice for last month",
        "System showing error 404",
        "My account has been hacked",
        "Change my subscription plan"
    ],
    "category": [
        "Billing","Account","Technical","General",
        "Billing","Technical","Account","Delivery",
        "Billing","Technical","Account","Billing"
    ],
    "priority": [
        "High","Medium","High","Low",
        "High","High","Medium","Medium",
        "Low","High","High","Low"
    ]
}

df = pd.DataFrame(data)

# ===============================
# 3. Text Preprocessing Function
# ===============================

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

df["clean_text"] = df["ticket_text"].apply(preprocess_text)

# ===============================
# 4. Feature Extraction (TF-IDF)
# ===============================

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])

y_category = df["category"]
y_priority = df["priority"]

# ===============================
# 5. Train-Test Split
# ===============================

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X, y_category, test_size=0.3, random_state=42
)

X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
    X, y_priority, test_size=0.3, random_state=42
)

# ===============================
# 6. Train Models
# ===============================

category_model = MultinomialNB()
category_model.fit(X_train_cat, y_train_cat)

priority_model = LogisticRegression()
priority_model.fit(X_train_pri, y_train_pri)

# ===============================
# 7. Model Evaluation
# ===============================

y_pred_cat = category_model.predict(X_test_cat)
print("===== CATEGORY MODEL =====")
print("Accuracy:", accuracy_score(y_test_cat, y_pred_cat))
print(classification_report(y_test_cat, y_pred_cat))
print("Confusion Matrix:\n", confusion_matrix(y_test_cat, y_pred_cat))

print("\n")

y_pred_pri = priority_model.predict(X_test_pri)
print("===== PRIORITY MODEL =====")
print("Accuracy:", accuracy_score(y_test_pri, y_pred_pri))
print(classification_report(y_test_pri, y_pred_pri))
print("Confusion Matrix:\n", confusion_matrix(y_test_pri, y_pred_pri))

# ===============================
# 8. Prediction Function
# ===============================

def predict_ticket(ticket):
    cleaned = preprocess_text(ticket)
    vector = vectorizer.transform([cleaned])
    
    predicted_category = category_model.predict(vector)[0]
    predicted_priority = priority_model.predict(vector)[0]
    
    print("\nTicket:", ticket)
    print("Predicted Category:", predicted_category)
    print("Predicted Priority:", predicted_priority)
    print("My ticket success")

# Example
predict_ticket("My account has been hacked and I cannot login")
predict_ticket("I want refund for my payment")
predict_ticket("Website is not working properly")
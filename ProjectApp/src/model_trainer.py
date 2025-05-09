# src/model_trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from dotenv import load_dotenv
import os

load_dotenv()
# Load the data
df = pd.read_csv(os.getenv("cleaned_csv_with_text_path"))

# Generate sentiment label heuristically (positive if satisfaction is high)
df['label'] = df['local_service_satisfaction'].apply(lambda x: 1 if x in ['Very Satisfied', 'Satisfied', 'Good', 'Very Good', 'Excellent', 'Fair'] else 0)

# Features and target
X = df['clean_text']
y = df['label']

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train Logistic Regression
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train, y_train)
log_preds = log_reg.predict(X_test)
print("\nLogistic Regression Report:\n", classification_report(y_test, log_preds))

# Train SVM
svm = SVC(kernel='linear', class_weight='balanced')
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
print("\nSVM Report:\n", classification_report(y_test, svm_preds))

# Save models
joblib.dump(log_reg, "D:/AMITY/Semester_4/5. Major Project/mba-semester4-major-project-qollabb/ProjectApp/models/logistic_model.pkl")
joblib.dump(svm, "D:/AMITY/Semester_4/5. Major Project/mba-semester4-major-project-qollabb/ProjectApp/models/svm_model.pkl")
joblib.dump(vectorizer, "D:/AMITY/Semester_4/5. Major Project/mba-semester4-major-project-qollabb/ProjectApp/models/tfidf_vectorizer.pkl")
print("\nModels and vectorizer saved to /models")

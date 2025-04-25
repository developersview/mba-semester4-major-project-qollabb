# src/nlp_pipeline.py

import spacy
import string
from sklearn.base import BaseEstimator, TransformerMixin

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.punctuations = string.punctuation
        self.stopwords = nlp.Defaults.stop_words

    def clean_text(self, text):
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc 
                  if token.lemma_ not in self.stopwords 
                  and token.lemma_ not in self.punctuations
                  and token.is_alpha]
        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(self.clean_text)

# Utility function to run pipeline
if __name__ == '__main__':
    import pandas as pd

    path = "D:/AMITY/Semester_4/5. Major Project/mba-semester4-major-project-qollabb/ProjectApp/data/preprocessed/cleaned_citizen_feedback.csv"
    df = pd.read_csv(path)

    # Combine all feedback columns into one
    df['full_feedback'] = df[['transport_suggestions', 'park_suggestions', 'library_suggestions', 'local_service_suggestions', 'local_service_satisfaction']].fillna('').agg(' '.join, axis=1)

    processor = SpacyPreprocessor()
    df['clean_text'] = processor.transform(df['full_feedback'])

    df.to_csv("D:/AMITY/Semester_4/5. Major Project/mba-semester4-major-project-qollabb/ProjectApp/data/processed/cleaned_feedback_with_text.csv", index=False)
    print("Cleaned feedback with NLP text saved to processed directory.")

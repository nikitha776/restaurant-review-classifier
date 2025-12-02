"""Utility script to (re)train the sentiment model with the local environment.

Running this ensures the pickled artifacts are compatible with the installed
scikit-learn version, avoiding runtime mismatches when loading in FastAPI.
"""
from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Add project root to path to import app.text_utils
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.text_utils import preprocess_text_with_negation

DATA_PATH = ROOT / "Restaurant_Reviews.tsv"
MODEL_PATH = ROOT / "model.pkl"
VECTORIZER_PATH = ROOT / "vectorizer.pkl"


def train() -> None:
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, delimiter="\t", quoting=3)
    
    # Preprocess data BEFORE splitting/vectorizing
    print("Preprocessing text (this might take a moment)...")
    # Ensure text is string
    df['Review'] = df['Review'].astype(str)
    X_processed = df['Review'].map(preprocess_text_with_negation)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, df["Liked"], random_state=0, test_size=0.2, stratify=df["Liked"]
    )

    print("Vectorizing...")
    # Parameters from restaurant_review_classifier2.ipynb
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training MultinomialNB...")
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)

    y_pred = classifier.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.3f}")

    # Save artifacts
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    train()

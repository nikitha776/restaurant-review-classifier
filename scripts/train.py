"""Utility script to (re)train the sentiment model with the local environment.

Running this ensures the pickled artifacts are compatible with the installed
scikit-learn version, avoiding runtime mismatches when loading in FastAPI.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Restaurant_Reviews.tsv"
MODEL_PATH = ROOT / "model.pkl"
VECTORIZER_PATH = ROOT / "vectorizer.pkl"


def train() -> None:
    df = pd.read_csv(DATA_PATH, delimiter="\t", quoting=3)
    X_train, X_test, y_train, y_test = train_test_split(
        df["Review"], df["Liked"], random_state=0, test_size=0.2
    )

    vectorizer = TfidfVectorizer()
    classifier = BernoulliNB()

    clf = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.3f}")

    joblib.dump(clf.named_steps["classifier"], MODEL_PATH)
    joblib.dump(clf.named_steps["vectorizer"], VECTORIZER_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    train()

"""
Model training pipeline.
Trains Logistic Regression and Multinomial Naive Bayes,
compares them, and saves the best model + vectorizer.
"""

import os
import sys
import logging

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Add project root to path so we can import app.preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.preprocessing import clean_text

# ──────────────── Config ────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Load and preprocess the dataset."""
    logger.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Class distribution:\n%s", df["label"].value_counts().to_string())
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def train_and_evaluate():
    """Full training pipeline."""
    df = load_data()

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train size: %d | Test size: %d", X_train.shape[0], X_test.shape[0])

    # ── Models ──
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0, solver="lbfgs",
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
    }

    results = {}
    for name, model in models.items():
        logger.info("Training %s …", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cv_scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="accuracy")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

        logger.info("── %s ──", name)
        logger.info("  Accuracy : %.4f", acc)
        logger.info("  Precision: %.4f", prec)
        logger.info("  Recall   : %.4f", rec)
        logger.info("  F1-score : %.4f", f1)
        logger.info("  CV Accuracy: %.4f (±%.4f)", cv_scores.mean(), cv_scores.std())
        print(f"\n{'='*50}")
        print(f"Classification Report — {name}")
        print("=" * 50)
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print()

    # ── Select best model ──
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    logger.info("✅ Best model: %s (F1=%.4f)", best_name, results[best_name]["f1"])

    # ── Save ──
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Vectorizer saved to %s", VECTORIZER_PATH)

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>7} {'CV Mean':>8}")
    print("-" * 60)
    for name, r in results.items():
        marker = " ★" if name == best_name else ""
        print(
            f"{name:<30} {r['accuracy']:>8.4f} {r['precision']:>9.4f} "
            f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['cv_mean']:>8.4f}{marker}"
        )
    print("=" * 60)

    return best_name, results


if __name__ == "__main__":
    train_and_evaluate()

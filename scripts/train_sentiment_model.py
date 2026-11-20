#!/usr/bin/env python3
"""
Utility script to train and evaluate the sentiment analysis logistic regression model.

The implementation mirrors the original hackathon notebook while adding a few
quality-of-life improvements for reproducibility (argument parsing, path handling,
and plot exports).
"""
from __future__ import annotations

import argparse
import re
import string
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "Reviews" / "data_classification_80.csv"
DEFAULT_EXTRA_PATH = PROJECT_ROOT / "Reviews" / "data_classification_20.csv"

# These objects are immutable for the lifetime of a run.  Initialising them
# once avoids rebuilding them for every review in the 344k-row training set.
STOP_WORDS: set[str] | None = None
STEMMER: SnowballStemmer | None = None


def ensure_nltk_resources(resources: Iterable[str]) -> None:
    """Ensure the required NLTK resources are available locally."""
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe supporting both CSV and Excel sources."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required to read Excel files. "
                "Install it or convert the dataset to CSV."
            ) from exc

    raise ValueError(f"Unsupported file format: {path.suffix}")


def preprocess_text(text: str) -> str:
    """Clean and normalize a review before vectorization."""
    if pd.isnull(text):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+|@\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)

    global STOP_WORDS, STEMMER
    if STOP_WORDS is None:
        STOP_WORDS = set(stopwords.words("english"))
    if STEMMER is None:
        STEMMER = SnowballStemmer("english")

    words = [word for word in text.split() if word not in STOP_WORDS]
    words = [STEMMER.stem(word) for word in words]

    return " ".join(words)


def vectorize_text(train_series: pd.Series, test_series: pd.Series, max_features: int) -> tuple:
    """Fit a TF-IDF vectorizer on the training data and transform both splits."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_vect = vectorizer.fit_transform(train_series)
    X_test_vect = vectorizer.transform(test_series)
    return vectorizer, X_train_vect, X_test_vect


def train_model(X_train, y_train) -> LogisticRegression:
    """Train the logistic regression model with class balancing."""
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def plot_roc_curve(y_true, y_scores, output_path: Path) -> None:
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    """Plot and save the confusion matrix using the conventional layout."""
    cm = confusion_matrix(y_true, y_pred)
    labels = np.array([["TN", "FP"], ["FN", "TP"]])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    cmap = colors.ListedColormap(["#e0f3ff", "#08519c"])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{labels[i, j]}\n{cm[i, j]}", ha="center", va="center", color=color)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def preprocess_dataframe(df: pd.DataFrame) -> pd.Series:
    """Apply the text preprocessing pipeline to the `Review` column."""
    if "Review" not in df.columns:
        raise KeyError("Expected column 'Review' in dataset.")
    return df["Review"].apply(preprocess_text)


def evaluate_dataset(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    df: pd.DataFrame,
    label_column: str = "Promotore",
) -> float:
    """Evaluate the trained model on an additional dataset."""
    if label_column not in df.columns:
        raise KeyError(f"Expected column '{label_column}' in dataset.")

    cleaned = preprocess_dataframe(df)
    X_vect = vectorizer.transform(cleaned)
    y_true = df[label_column]
    y_pred = model.predict(X_vect)
    return accuracy_score(y_true, y_pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the sentiment analysis model.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the main training dataset (CSV or XLSX).",
    )
    parser.add_argument(
        "--extra-path",
        type=Path,
        default=DEFAULT_EXTRA_PATH,
        help="Optional extra dataset for post-training evaluation.",
    )
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "roc_curve.png",
        help="File path where the ROC curve image will be saved.",
    )
    parser.add_argument(
        "--confusion-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "confusion_matrix.png",
        help="File path where the confusion matrix image will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to allocate to the validation split.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20000,
        help="Maximum number of TF-IDF features.",
    )
    parser.add_argument(
        "--skip-extra",
        action="store_true",
        help="Skip evaluation on the optional extra dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Tokenisation is performed with ``str.split`` below, so the project only
    # needs the English stop-word corpus.  Avoid downloading unused resources
    # at runtime: it keeps the command deterministic on fresh environments.
    ensure_nltk_resources(["stopwords"])

    train_df = load_dataframe(args.train_path)
    print(f"Loaded training dataset with {len(train_df)} rows from {args.train_path}")

    cleaned_reviews = preprocess_dataframe(train_df)
    y = train_df["Promotore"]

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_reviews, y, test_size=args.test_size, random_state=42, stratify=y
    )

    vectorizer, X_train_vect, X_test_vect = vectorize_text(
        X_train, X_test, max_features=args.max_features
    )

    model = train_model(X_train_vect, y_train)

    y_test_pred = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Validation accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred))

    y_test_proba = model.predict_proba(X_test_vect)[:, 1]
    plot_roc_curve(y_test, y_test_proba, args.roc_output)
    print(f"ROC curve saved to {args.roc_output}")

    plot_confusion_matrix(y_test, y_test_pred, args.confusion_output)
    print(f"Confusion matrix saved to {args.confusion_output}")

    if not args.skip_extra and args.extra_path.exists():
        extra_df = load_dataframe(args.extra_path)
        extra_accuracy = evaluate_dataset(model, vectorizer, extra_df)
        print(f"Extra dataset accuracy: {extra_accuracy:.4f}")
    elif not args.skip_extra:
        print(f"Extra dataset not found at {args.extra_path}. Skipping evaluation.")


if __name__ == "__main__":
    main()

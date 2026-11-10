"""Train the optional emotion model used by the chatbot.

Expects a text file with one `text;label` pair per line, e.g. the
"Emotions dataset for NLP" (labels: joy, sadness, anger, fear, love, surprise).

    python train.py --data data/train.txt

Writes models/emotion_model.joblib plus an evaluation report in reports/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=BASE_DIR / "data" / "train.txt", help="path to text;label file")
    p.add_argument("--sep", default=";", help="column separator (default ';')")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--out", type=Path, default=BASE_DIR / "models" / "emotion_model.joblib")
    p.add_argument("--reports", type=Path, default=BASE_DIR / "reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}\nSee README.md > 'Training the model'.")

    df = pd.read_csv(args.data, sep=args.sep, names=["text", "label"], engine="python").dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"] != ""].drop_duplicates("text")
    print(f"{len(df)} samples, {df['label'].nunique()} classes")
    print(df["label"].value_counts().to_string(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size, random_state=42, stratify=df["label"]
    )

    # TF-IDF handles lowercasing/tokenising; bigrams keep cues like "not happy".
    # Stop words are kept on purpose: negations carry the emotion.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, C=10, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classes": list(pipeline.classes_),
    }
    print(classification_report(y_test, y_pred, digits=3))
    print(f"accuracy={metrics['accuracy']}  macro-F1={metrics['macro_f1']}")

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", xticks_rotation=45)
        disp.ax_.set_title("Confusion matrix (held-out test set)")
        plt.tight_layout()
        plt.savefig(args.reports / "confusion_matrix.png", dpi=150)
        plt.close()
    except ImportError:
        print("matplotlib not installed - skipping confusion matrix plot")

    # Refit on everything for the model that ships; the metrics above are from the held-out split.
    pipeline.fit(df["text"], df["label"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.out)
    print(f"saved model -> {args.out}")


if __name__ == "__main__":
    main()

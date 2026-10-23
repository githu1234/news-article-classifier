"""Train and persist the BBC news category classifier."""
from __future__ import annotations
import pickle
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "bbc_text_cls.csv"
MODEL_PATH = APP_DIR / "data" / "news_classifier.pkl"

def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_df=0.95, sublinear_tf=True)),
        ("classifier", MultinomialNB(alpha=0.2)),
    ])

def train_and_save_model() -> dict:
    """Train on the bundled CSV and pickle the fitted pipeline and metadata."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data was not found: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    if not {"text", "labels"}.issubset(frame.columns):
        raise ValueError("The CSV must contain 'text' and 'labels' columns.")
    frame = frame.dropna(subset=["text", "labels"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["labels"] = frame["labels"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["labels"] != "")]
    if frame["labels"].nunique() < 2:
        raise ValueError("Training data needs at least two news categories.")
    x_train, x_valid, y_train, y_valid = train_test_split(
        frame["text"], frame["labels"], test_size=0.2, random_state=42,
        stratify=frame["labels"])
    validator = make_pipeline()
    validator.fit(x_train, y_train)
    accuracy = accuracy_score(y_valid, validator.predict(x_valid))
    model = make_pipeline()
    model.fit(frame["text"], frame["labels"])
    artifact = {
        "model": model,
        "categories": sorted(frame["labels"].unique().tolist()),
        "validation_accuracy": float(accuracy),
        "training_rows": int(len(frame)),
        "category_counts": {
            str(label): int(count)
            for label, count in frame["labels"].value_counts().sort_index().items()
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)
    return artifact

if __name__ == "__main__":
    result = train_and_save_model()
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Validation accuracy: {result['validation_accuracy']:.1%}")
    print(f"Training articles: {result['training_rows']}")

"""Streamlit app for classifying news topics with Naive Bayes."""
from __future__ import annotations
import pickle
from pathlib import Path
import pandas as pd
import streamlit as st
from train_model import DATA_PATH, MODEL_PATH, train_and_save_model

APP_DIR = Path(__file__).resolve().parent
PUBLIC_MODEL_PATH = APP_DIR / "news_classifier.pkl"

st.set_page_config(page_title="ArticleLens | News Topic Classifier", page_icon="📰",
                   layout="wide", initial_sidebar_state="expanded")

@st.cache_resource(show_spinner=False)
def load_artifact() -> dict:
    """Load a saved pickle or train once if it is missing or stale."""
    model_path = next((path for path in (MODEL_PATH, PUBLIC_MODEL_PATH) if path.exists()), None)
    needs_training = model_path is None
    if DATA_PATH.exists() and model_path is not None:
        needs_training = DATA_PATH.stat().st_mtime > model_path.stat().st_mtime
    if needs_training:
        return train_and_save_model()
    try:
        with model_path.open("rb") as file:
            artifact = pickle.load(file)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError("The saved model file has an unexpected format.")
        return artifact
    except (OSError, EOFError, pickle.UnpicklingError, ValueError):
        return train_and_save_model()

def category_name(label: str) -> str:
    return str(label).replace("_", " ").title()

st.markdown("""
<style>
.stApp { background: #f4f7fb; }
[data-testid="stSidebar"] { background: #101e35; }
[data-testid="stSidebar"] * { color: #edf3fb; }
.hero { padding: 1.8rem 2rem; border-radius: 20px; margin: .2rem 0 1.3rem;
 color: white; background: linear-gradient(115deg,#102544,#1d4770 70%,#2e7892); }
.hero h1 { margin: 0 0 .4rem; font-size: 2.2rem; }
.hero p { color: #d9e8f5; margin: 0; font-size: 1.03rem; }
.result-card { background: white; border: 1px solid #e5ebf2; border-radius: 16px;
 padding: 1.3rem 1.5rem; box-shadow: 0 8px 28px rgba(25,48,77,.06); }
.eyebrow { color: #61758a; font-size: .8rem; text-transform: uppercase;
 letter-spacing: .1em; font-weight: 700; }
.category { color: #173a5e; font-size: 2rem; font-weight: 750; margin: .2rem 0; }
.subtle { color: #61758a; }
div[data-testid="stMetric"] { background: white; border: 1px solid #e5ebf2;
 border-radius: 14px; padding: .8rem 1rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📰 ArticleLens")
    st.caption("News topic classifier")
    st.divider()
    with st.spinner("Loading the classifier…"):
        try:
            artifact = load_artifact()
        except Exception as error:
            st.error(f"Could not prepare the model: {error}")
            st.stop()
    st.markdown("### Model snapshot")
    st.metric("Training articles", f"{artifact.get('training_rows', 0):,}")
    accuracy = artifact.get("validation_accuracy")
    st.metric("Holdout accuracy", f"{accuracy:.1%}" if accuracy is not None else "—")
    st.caption("Naive Bayes · TF-IDF word-presence features")
    if artifact.get("category_counts"):
        st.markdown("### Topics in the training set")
        for label, count in artifact["category_counts"].items():
            st.write(f"**{category_name(label)}** · {count:,}")

st.markdown("""
<div class="hero">
<h1>News Article Topic Classifier</h1>
<p>Paste an article to see which topic it is most likely to discuss.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.2, 0.8], gap="large")
with left:
    st.subheader("Classify an article")
    headline = st.text_input("Headline (optional)", placeholder="Add a headline")
    article = st.text_area("Article text", height=310,
                           placeholder="Paste the article text here…",
                           help="A few sentences usually give the model more context.")
    classify = st.button("Classify article", type="primary", use_container_width=True)
with right:
    st.subheader("How classification works")
    st.markdown(
        "The app turns article text into TF-IDF features, then uses a **Naive Bayes** "
        "classifier to estimate the topic. The model scores the words and short phrases "
        "that appear in the article."
    )
    st.info(
        "Topic scores show how the model ranks the possible categories. They can help "
        "compare topics, but they are not guaranteed to be calibrated probabilities."
    )

if classify:
    content = " ".join(part.strip() for part in (headline, article) if part.strip())
    if len(content) < 30:
        st.warning("Add a little more text so the model has enough context to classify it.")
    else:
        model = artifact["model"]
        prediction = model.predict([content])[0]
        probabilities = model.predict_proba([content])[0]
        classes = model.named_steps["classifier"].classes_
        scores = pd.DataFrame({
            "Topic": [category_name(label) for label in classes],
            "Score": probabilities,
        }).sort_values("Score", ascending=False)
        st.markdown("### Classification result")
        result_col, score_col = st.columns([1.4, 1])
        with result_col:
            st.markdown(
                f'<div class="result-card"><div class="eyebrow">Predicted topic</div>'
                f'<div class="category">{category_name(prediction)}</div>'
                f'<div class="subtle">Highest model score</div></div>',
                unsafe_allow_html=True)
        with score_col:
            st.metric("Top topic score", f"{float(scores.iloc[0]['Score']):.1%}")
        st.markdown("#### Scores by topic")
        st.bar_chart(scores.set_index("Topic")[["Score"]])

st.caption(
    f"Training set: {artifact.get('training_rows', 0):,} labeled news articles. "
    "The source CSV is not included in the public repository."
)

# ArticleLens — News Topic Classifier

A Streamlit app that predicts the topic of a news article using a Naive Bayes classifier and TF-IDF word features. The trained model and its metadata are stored in a Python pickle.

## Run locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

The app loads `data/news_classifier.pkl` locally. If no model is available, it trains from `data/bbc_text_cls.csv`, which must contain `text` and `labels` columns.

## Public deployment

This public repository includes the app, a trained model pickle, and the dependencies. The original CSV is excluded. The public app loads the root-level `news_classifier.pkl`; a locally trained model in `data/news_classifier.pkl` takes precedence.

Deploy through Streamlit Community Cloud by selecting this repository, the `main` branch, and `app.py` as the entry point. Select Python 3.14 to match the runtime used to create the pickle. Set app visibility to public to share the `streamlit.app` URL.

## Model notes

- The classifier is Bernoulli Naive Bayes. It treats TF-IDF features as word-presence indicators.
- Accuracy is measured on a stratified 20% holdout split; the saved model is then fitted on the full dataset.
- Model scores may not be calibrated probabilities.
- The app loads only the included pickle or a model created locally. Pickle files can execute code; do not use one from an untrusted source.
- To retrain, put the source CSV at `data/bbc_text_cls.csv` and run `python train_model.py`.

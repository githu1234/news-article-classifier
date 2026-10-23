# News article classifier

A Streamlit app that classifies article text into a BBC news category using TF-IDF features and the Multinomial Naive Bayes algorithm. The fitted pipeline and metadata are saved with Python pickle.

## Run locally

Install dependencies: python -m pip install -r requirements.txt

Start the app: streamlit run app.py

Locally, the app loads data/news_classifier.pkl. If it is missing, the app trains from data/bbc_text_cls.csv. The CSV has text and labels columns.

## Public deployment

This public repository includes the app, a trained model pickle, and requirements. The original CSV is excluded so raw articles are not published. The public app loads the root-level news_classifier.pkl; a locally trained model in data/news_classifier.pkl takes precedence.

Deploy through Streamlit Community Cloud by selecting this repository, the main branch, and app.py as the entrypoint. Select Python 3.14 in Advanced settings to match the Python version used to create the pickle. Set the app visibility to public to share its streamlit.app URL.

## Model notes

- Accuracy is measured on a stratified 20% holdout split; the saved model is then fitted on the full dataset.
- Confidence scores are model scores and may not be calibrated probabilities.
- The app loads only the included pickle or a model created locally. Pickle files can execute code; do not use a pickle from an untrusted source.
- To retrain, put the CSV at data/bbc_text_cls.csv and run python train_model.py. The app will use the newer local model automatically.

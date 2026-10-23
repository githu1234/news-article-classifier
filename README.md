# News article classifier

A Streamlit app that classifies article text into a BBC news category using TF-IDF features and the Multinomial Naive Bayes algorithm. The fitted pipeline and metadata are saved locally with Python pickle.

## Run locally

Install dependencies: python -m pip install -r requirements.txt

Start the app: streamlit run app.py

The app loads data/news_classifier.pkl. If the model is missing, it trains from data/bbc_text_cls.csv and saves a new pickle. The supplied CSV has text and labels columns.

## Public deployment

This folder is ready to publish to a public GitHub repository and deploy from Streamlit Community Cloud. The saved model is included; the original CSV is excluded by .gitignore so the raw articles are not published. Streamlit uses the repository's requirements.txt to install dependencies.

After publishing the repository, sign in to Streamlit Community Cloud with GitHub, create an app from the repository, choose app.py as the entrypoint, and set the app visibility to public. Community Cloud will provide a shareable streamlit.app URL.

## Model notes

- The displayed holdout accuracy is measured on a stratified 20% split; the saved model is then fitted on the full dataset.
- Confidence scores are not guaranteed to be calibrated probabilities.
- Pickle files can execute code when loaded. Only use the model file included with this app or one you created yourself.
- To retrain, put the CSV at data/bbc_text_cls.csv and run python train_model.py. The app will also retrain automatically if the CSV is newer than the pickle.

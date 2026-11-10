# Emotion-Aware Chatbot

A small Flask chatbot that detects the emotion in a message and replies with a
supportive, emotion-specific response. It recognises 20+ emotions (happy, sad,
anxious, lonely, proud, grateful, ...) and shows the detected emotion next to
each reply.

## How it works

Each message goes through three layers, in order (`emotion.py`):

| Layer | What it does | When it is used |
|---|---|---|
| Keyword lexicon | Matches explicit emotion words and phrases, with simple negation handling ("not happy") | Message names a feeling |
| Trained model *(optional)* | TF-IDF (1–2 grams) + Logistic Regression | `models/emotion_model.joblib` exists and the model is confident |
| VADER sentiment | Overall polarity → happy / sad / neutral | Fallback |

Messages that mention self-harm are caught first and always get a reply pointing
to real-world support.

Replies live in `responses.json`, so they can be edited without touching code.

## Project structure

```
app.py              Flask app and routes
emotion.py          Emotion detection (lexicon, model, VADER)
train.py            Trains and evaluates the optional ML model
responses.json      Reply for each emotion
templates/          index.html
static/             style.css, app.js
tests/              pytest suite
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5001. Set `PORT` to change the port and `FLASK_DEBUG=1`
for auto-reload.

## Training the model (optional)

The app works without a trained model. To add one, download a `text;label`
dataset such as the
[Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
and save its `train.txt` as `data/train.txt`, then:

```bash
python train.py --data data/train.txt
```

This prints a classification report, writes `reports/metrics.json` and
`reports/confusion_matrix.png`, and saves `models/emotion_model.joblib`.
Restart the app to load it; `GET /health` reports whether a model is loaded.

Dataset labels are mapped to reply keys in `MODEL_LABEL_MAP` in `emotion.py`.

## API

`POST /get` with JSON `{"msg": "I feel lonely"}` (form data also works):

```json
{
  "emotion": "lonely",
  "confidence": 1.0,
  "method": "lexicon",
  "sentiment": -0.361,
  "response": "Feeling lonely can be tough. ..."
}
```

## Tests

```bash
pytest
```

## Limitations

This is a learning project, not a mental-health tool. The lexicon is small and
hand-written, replies are fixed templates, and there is no conversation memory.

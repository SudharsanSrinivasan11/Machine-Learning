"""Emotion detection for the chatbot.

Three layers, tried in order:

1. Keyword lexicon  - explicit emotion words ("I feel lonely") for the 20+
                      fine-grained emotions in responses.json.
2. Trained model    - TF-IDF + Logistic Regression produced by train.py
                      (optional; used only if models/emotion_model.joblib exists).
3. VADER sentiment  - overall polarity as a last resort (happy / sad / neutral).
"""
from __future__ import annotations

import re
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "emotion_model.joblib"

# Minimum model probability before we trust its prediction.
MODEL_CONFIDENCE_THRESHOLD = 0.45
# VADER compound score beyond which text counts as clearly positive/negative.
POLARITY_THRESHOLD = 0.05

# Labels of the common "Emotions dataset for NLP" mapped to responses.json keys.
MODEL_LABEL_MAP = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fearful",
    "surprise": "surprised",
    "love": "loving",
}

LEXICON: dict[str, list[str]] = {
    "happy": ["happy", "glad", "joyful", "joy", "delighted", "cheerful", "pleased"],
    "sad": ["sad", "unhappy", "depressed", "miserable", "heartbroken", "crying", "upset", "feeling down", "feeling low"],
    "angry": ["angry", "mad", "furious", "annoyed", "irritated", "pissed", "outraged", "frustrated", "hate"],
    "surprised": ["surprised", "shocked", "astonished", "amazed", "stunned", "unexpected", "unbelievable"],
    "fearful": ["afraid", "scared", "terrified", "frightened", "fearful", "fear", "petrified"],
    "disgusted": ["disgusted", "disgusting", "gross", "revolting", "repulsed", "sickening", "nasty"],
    "confused": ["confused", "confusing", "puzzled", "feeling lost", "unsure", "baffled", "perplexed"],
    "bored": ["bored", "boring", "dull", "tedious", "uninterested"],
    "excited": ["excited", "thrilled", "pumped", "eager", "ecstatic", "stoked"],
    "anxious": ["anxious", "anxiety", "nervous", "worried", "stressed", "tense", "panicking", "overwhelmed", "uneasy"],
    "ashamed": ["ashamed", "embarrassed", "humiliated", "shame", "mortified"],
    "jealous": ["jealous", "envious", "envy", "jealousy"],
    "guilty": ["guilty", "guilt", "regret", "remorse", "regretful"],
    "proud": ["proud", "accomplished", "achieved", "pride"],
    "hopeful": ["hopeful", "optimistic", "hoping", "hope", "looking forward"],
    "lonely": ["lonely", "alone", "isolated", "lonesome", "left out", "no friends"],
    "disappointed": ["disappointed", "disappointing", "let down", "disheartened"],
    "grateful": ["grateful", "thankful", "blessed", "appreciate", "appreciative", "thanks", "thank you"],
    "curious": ["curious", "wondering", "intrigued", "wonder"],
    "relieved": ["relieved", "relief", "phew", "finally over"],
    "loving": ["love", "adore", "loved", "affection", "cherish"],
}

# Checked before anything else so a serious message never gets a generic reply.
CRISIS_PATTERN = re.compile(
    r"\b(kill(ing)? myself|suicid\w*|end(ing)? my life|want to die|wanna die|"
    r"self[- ]?harm\w*|hurt(ing)? myself|no reason to live|better off dead)\b",
    re.IGNORECASE,
)

NEGATORS = {"not", "no", "never", "dont", "don't", "isnt", "isn't", "wasnt", "wasn't",
            "cant", "can't", "wont", "won't", "aint", "ain't", "hardly", "barely", "without"}
NEGATION_WINDOW = 3  # tokens before an emotion word that can negate it

_TOKEN_RE = re.compile(r"[a-z']+")


def _build_lookup() -> tuple[dict[str, str], dict[tuple[str, ...], str]]:
    words, phrases = {}, {}
    for emotion, terms in LEXICON.items():
        for term in terms:
            parts = tuple(term.split())
            if len(parts) == 1:
                words.setdefault(term, emotion)
            else:
                phrases[parts] = emotion
    return words, phrases


_WORDS, _PHRASES = _build_lookup()


class EmotionClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.vader = SentimentIntensityAnalyzer()
        self.model = None
        if model_path.exists():
            try:
                import joblib

                self.model = joblib.load(model_path)
            except Exception as exc:  # corrupt / incompatible model: keep running without it
                print(f"[emotion] could not load {model_path.name}: {exc}")

    @property
    def has_model(self) -> bool:
        return self.model is not None

    def _lexicon_scores(self, text: str) -> dict[str, int]:
        tokens = _TOKEN_RE.findall(text.lower())
        scores: dict[str, int] = {}

        def negated(i: int) -> bool:
            return any(t in NEGATORS for t in tokens[max(0, i - NEGATION_WINDOW):i])

        for i, tok in enumerate(tokens):
            emotion = _WORDS.get(tok)
            if emotion and not negated(i):
                scores[emotion] = scores.get(emotion, 0) + 1
        for parts, emotion in _PHRASES.items():
            n = len(parts)
            for i in range(len(tokens) - n + 1):
                if tuple(tokens[i:i + n]) == parts and not negated(i):
                    scores[emotion] = scores.get(emotion, 0) + 2  # phrases are more specific
        return scores

    def classify(self, text: str) -> dict:
        """Return {"emotion", "confidence", "method", "sentiment"} for a message."""
        compound = self.vader.polarity_scores(text)["compound"]
        result = {"sentiment": round(compound, 3)}

        if CRISIS_PATTERN.search(text):
            return {**result, "emotion": "crisis", "confidence": 1.0, "method": "safety"}

        scores = self._lexicon_scores(text)
        if scores:
            emotion = max(scores, key=scores.get)
            confidence = scores[emotion] / sum(scores.values())
            return {**result, "emotion": emotion, "confidence": round(confidence, 2), "method": "lexicon"}

        if self.model is not None:
            probs = self.model.predict_proba([text])[0]
            best = int(probs.argmax())
            if probs[best] >= MODEL_CONFIDENCE_THRESHOLD:
                label = str(self.model.classes_[best])
                return {**result, "emotion": MODEL_LABEL_MAP.get(label, label),
                        "confidence": round(float(probs[best]), 2), "method": "model"}

        if compound >= POLARITY_THRESHOLD:
            emotion = "happy"
        elif compound <= -POLARITY_THRESHOLD:
            emotion = "sad"
        else:
            emotion = "neutral"
        return {**result, "emotion": emotion, "confidence": round(abs(compound), 2), "method": "sentiment"}

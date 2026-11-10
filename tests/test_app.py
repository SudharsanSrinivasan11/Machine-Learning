import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import RESPONSES, app  # noqa: E402
from emotion import LEXICON, MODEL_LABEL_MAP, EmotionClassifier  # noqa: E402


@pytest.fixture(scope="module")
def clf():
    # Point at a missing path so tests exercise the rule-based layers deterministically.
    return EmotionClassifier(model_path=ROOT / "does-not-exist.joblib")


@pytest.fixture()
def client():
    return app.test_client()


@pytest.mark.parametrize("text,expected", [
    ("I feel so lonely these days", "lonely"),
    ("I'm really nervous about my exam", "anxious"),
    ("I am so proud of my sister", "proud"),
    ("thank you so much for the help", "grateful"),
    ("this lecture is boring", "bored"),
    ("I am not happy with this at all", "sad"),       # negation flips to polarity fallback
    ("What a wonderful, fantastic day", "happy"),      # no lexicon hit -> VADER
    ("The meeting is at 3 pm", "neutral"),
    ("sometimes I want to die", "crisis"),
])
def test_classify(clf, text, expected):
    assert clf.classify(text)["emotion"] == expected


def test_every_emotion_has_a_response():
    emotions = set(LEXICON) | set(MODEL_LABEL_MAP.values()) | {"neutral", "crisis", "happy", "sad"}
    assert emotions <= set(RESPONSES)


def test_home_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"composer" in res.data


def test_chat_json_and_form(client):
    for kwargs in ({"json": {"msg": "I feel lonely"}}, {"data": {"msg": "I feel lonely"}}):
        body = client.post("/get", **kwargs).get_json()
        assert body["emotion"] == "lonely"
        assert body["response"] == RESPONSES["lonely"]


@pytest.mark.parametrize("msg", ["", "   ", "x" * 1001])
def test_rejects_bad_input(client, msg):
    assert client.post("/get", json={"msg": msg}).status_code == 400


def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"

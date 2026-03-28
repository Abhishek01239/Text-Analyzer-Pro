from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

TRAIN_DATA = [
    # Positive samples
    ("I love this product it is amazing and works perfectly", "POSITIVE"),
    ("Excellent quality and fast delivery highly recommended", "POSITIVE"),
    ("Best purchase I have ever made absolutely wonderful", "POSITIVE"),
    ("Great experience the staff was very helpful and friendly", "POSITIVE"),
    ("I am so happy with this item it exceeded my expectations", "POSITIVE"),
    ("Fantastic service and the product is top notch quality", "POSITIVE"),
    ("Very satisfied with my purchase will definitely buy again", "POSITIVE"),
    ("Outstanding performance and great value for money", "POSITIVE"),
    ("The movie was brilliant and the acting was superb", "POSITIVE"),
    ("I enjoyed every moment of this experience truly amazing", "POSITIVE"),
    ("Perfect product exactly as described fast shipping", "POSITIVE"),
    ("Highly recommend this to everyone great quality", "POSITIVE"),
    ("Wonderful experience from start to finish loved it", "POSITIVE"),
    ("This is the best thing I have bought in a long time", "POSITIVE"),
    ("Impressed with the quality and attention to detail", "POSITIVE"),
    ("Great customer service and very quick response time", "POSITIVE"),
    ("Love the design and functionality works like a charm", "POSITIVE"),
    ("Amazing value for money very happy with this purchase", "POSITIVE"),
    ("Superb quality and arrived earlier than expected", "POSITIVE"),
    ("Five stars excellent in every way very pleased", "POSITIVE"),

    # Negative samples
    ("This product is terrible complete waste of money", "NEGATIVE"),
    ("Worst experience ever I am very disappointed", "NEGATIVE"),
    ("Horrible quality broke after one day do not buy", "NEGATIVE"),
    ("Very unhappy with this purchase total garbage", "NEGATIVE"),
    ("Awful service rude staff and long waiting time", "NEGATIVE"),
    ("Do not buy this product it stopped working immediately", "NEGATIVE"),
    ("Completely useless and very poor quality material", "NEGATIVE"),
    ("I regret buying this it is just a waste of money", "NEGATIVE"),
    ("Terrible customer service they ignored my complaint", "NEGATIVE"),
    ("Disgusting quality nothing like the description", "NEGATIVE"),
    ("Very disappointed with the product not worth it", "NEGATIVE"),
    ("Broken on arrival and return process is a nightmare", "NEGATIVE"),
    ("Worst purchase of my life absolutely terrible", "NEGATIVE"),
    ("Poor quality and the delivery took forever", "NEGATIVE"),
    ("Not happy at all this product failed after one use", "NEGATIVE"),
    ("Dreadful experience would not recommend to anyone", "NEGATIVE"),
    ("Cheap and nasty material fell apart immediately", "NEGATIVE"),
    ("Scam product looks nothing like the photos avoid", "NEGATIVE"),
    ("Frustrating experience the app keeps crashing always", "NEGATIVE"),
    ("Zero stars if I could terrible in every single way", "NEGATIVE"),
]

def _build_model() -> Pipeline:
    texts = [t for t, _ in TRAIN_DATA]
    labels = [l for _, l in TRAIN_DATA]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            stop_words="english",
            max_features=5000
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42
        )),
    ])

    model.fit(texts, labels)

    return model

_MODEL = _build_model()

def analyze_sentiment(text: str) -> dict:
    """
    Returns:
        {
            "label" : "POSITIVE" or "NEGATIVE",
            "confidense" : float,
            "pos_score" : float,
            "neg_score" : float
        }
    """

    proba = _MODEL.predict_proba([text])[0]
    labels = _MODEL.classes_

    scores  = dict(zip(labels, proba))

    pos = scores.get("POSITIVE", 0.0)
    neg = scores.get("NEGATIVE", 0.0)

    label = "POSITIVE" if pos >= neg else "NEGATIVE"
    confidence = max(pos, neg)

    return {
        "label": label,
        "confidence" : confidence,
        "pos_score" : round(pos, 4),
        "neg_score": round(neg, 4)
    }
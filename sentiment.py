from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

# Training Data 
TRAIN_TEXTS = [
    # Positive
    "I love this product it is amazing and wonderful",
    "Excellent quality and great customer service highly recommended",
    "This is the best purchase I have ever made fantastic",
    "I am so happy with this absolutely brilliant and perfect",
    "Great experience wonderful service and superb quality loved it",
    "Amazing product works perfectly exceeded all my expectations",
    "Fantastic value for money very satisfied with my purchase",
    "Outstanding performance highly recommend this to everyone",
    "Very happy with the results beautiful and high quality",
    "Loved every bit of it incredible experience will buy again",
    "Superb quality fast delivery and excellent packaging great",
    "Best product ever bought very pleased and satisfied customer",
    "Wonderful item exactly as described very happy with purchase",
    "Great value highly satisfied recommend to all my friends",
    "Excellent product perfect condition arrived quickly very happy",
    "I really enjoyed this it was a great experience overall",
    "This works perfectly I am so impressed with the quality",
    "Amazing value for money very pleased with this product",
    "Highly recommend brilliant product exceeded my expectations",
    "Love it so much great quality and fast shipping thanks",

    # Negative
    "This product is terrible complete waste of money avoid",
    "Worst purchase ever made very disappointed and frustrated",
    "Horrible quality broke after one day terrible experience",
    "Completely useless does not work at all very angry",
    "Awful product poor quality not worth the money at all",
    "Disgusting customer service rude staff never coming back",
    "Very disappointed with this purchase total waste of money",
    "Terrible quality broken on arrival do not buy this",
    "Pathetic product failed to work extremely frustrated unhappy",
    "Worst experience ever horrible service and bad quality avoid",
    "Do not buy this it is a scam total disappointment",
    "Cheap and nasty broke immediately terrible customer support",
    "Useless product does not do what it says very bad",
    "Extremely disappointed poor performance and bad quality item",
    "Horrible experience rude service and defective product bad",
    "Very bad quality falling apart after one use terrible",
    "Disgusting quality not as described totally unacceptable bad",
    "Awful experience broken product and no refund very angry",
    "Terrible service long wait and rude staff very unhappy",
    "Worst product I have ever bought completely useless garbage",
]

TRAIN_LABELS = [1] * 20 + [0] * 20  # 1 = Positive, 0 = Negative

# Build and Train Pipeline 
_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=5000
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

_model.fit(TRAIN_TEXTS, TRAIN_LABELS)


# Public Function 
def analyze_sentiment(text: str) -> dict:
    """
    Predicts sentiment of the given text.

    Returns:
        {
          "label"     : "POSITIVE" or "NEGATIVE",
          "confidence": float,
          "pos_score" : float,
          "neg_score" : float
        }
    """
    proba     = _model.predict_proba([text])[0]
    neg_score = round(float(proba[0]), 4)
    pos_score = round(float(proba[1]), 4)

    if pos_score >= neg_score:
        label      = "POSITIVE"
        confidence = pos_score
    else:
        label      = "NEGATIVE"
        confidence = neg_score

    return {
        "label":      label,
        "confidence": confidence,
        "pos_score":  pos_score,
        "neg_score":  neg_score,
    }

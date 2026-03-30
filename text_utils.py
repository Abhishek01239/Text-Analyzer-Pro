import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Common stopwords to filter out 
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "my",
    "your", "his", "her", "its", "our", "their", "what", "which", "who",
    "not", "no", "so", "if", "as", "up", "out", "about", "into", "than",
    "then", "when", "there", "also", "just", "more", "very", "all", "any",
}


# 1. Keyword Extractor 
def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    Extracts top N keywords using TF-IDF on the input text.
    Returns list of [keyword, score] pairs sorted by importance.
    """
    # Split text into sentences to simulate multiple documents
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    if len(sentences) < 2:
        # Fallback: split into chunks of words
        words   = text.split()
        chunk   = max(1, len(words) // 3)
        sentences = [
            " ".join(words[i:i+chunk])
            for i in range(0, len(words), chunk)
        ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=500
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    # Average TF-IDF score across all sentences
    scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    # Sort by score descending
    sorted_indices = scores.argsort()[::-1]
    results = []
    for idx in sorted_indices:
        keyword = feature_names[idx]
        score   = round(float(scores[idx]), 4)
        if score > 0:
            results.append([keyword, score])
        if len(results) >= top_n:
            break

    return results


# 2. Text Statistics 
def get_text_stats(text: str) -> dict:
    """
    Returns detailed statistics about the input text.
    """
    # Words
    words        = text.split()
    total_words  = len(words)
    unique_words = len(set(w.lower().strip(".,!?\"'();:-") for w in words))

    # Sentences
    sentences     = re.split(r'[.!?]+', text)
    sentences     = [s.strip() for s in sentences if s.strip()]
    total_sentences = len(sentences)

    # Characters
    characters          = len(text)
    characters_no_spaces = len(text.replace(" ", ""))

    # Averages
    avg_word_length = round(
        sum(len(w.strip(".,!?\"'();:-")) for w in words) / max(total_words, 1), 1
    )
    avg_sentence_length = round(total_words / max(total_sentences, 1), 1)

    # Paragraphs
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])

    # Vocabulary richness
    vocabulary_richness = unique_words / max(total_words, 1)

    return {
        "total_words":          total_words,
        "unique_words":         unique_words,
        "sentences":            total_sentences,
        "characters":           characters,
        "characters_no_spaces": characters_no_spaces,
        "avg_word_length":      avg_word_length,
        "avg_sentence_length":  avg_sentence_length,
        "paragraphs":           paragraphs,
        "vocabulary_richness":  vocabulary_richness,
    }


# 3. Word Frequency 
def get_word_frequency(text: str, top_n: int = 15) -> list:
    """
    Returns top N most frequent words (excluding stopwords).
    Returns list of [word, count] pairs.
    """
    # Clean and tokenize
    cleaned = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    words   = cleaned.split()

    # Filter stopwords and short words
    filtered = [
        w for w in words
        if w not in STOPWORDS and len(w) > 2
    ]

    counter = Counter(filtered)
    top     = counter.most_common(top_n)

    return [[word, count] for word, count in top]

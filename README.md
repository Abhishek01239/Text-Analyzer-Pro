# 🔍 Text Analyzer Pro

A multi-feature NLP web app built with Streamlit + Scikit-learn.
No PyTorch. No HuggingFace. No errors. Just pure ML.

---

## ✨ Features

| Feature | Description | Library |
|---|---|---|
| 😊 Sentiment Analysis | Predicts Positive / Negative with confidence score | Scikit-learn |
| 🔑 Keyword Extractor | Extracts top keywords using TF-IDF | Scikit-learn |
| 📊 Text Statistics | Word count, sentences, readability stats | Pure Python |
| 🔤 Word Frequency | Top N most used words with bar chart | Collections |

---

## ⚙️ Setup & Run

### 1. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## 📁 Project Structure
```
text_analyzer/
├── app.py            # Main Streamlit UI (4 tabs)
├── sentiment.py      # Sentiment model (TF-IDF + Logistic Regression)
├── text_utils.py     # Keywords, Statistics, Word Frequency
├── requirements.txt  # Dependencies
└── README.md         # This file
```

---

## 📝 Notes

- **No model downloads** — sentiment model trains instantly on startup
- **No GPU needed** — runs on any basic laptop
- **Python 3.10 / 3.11 / 3.13** — works on all versions
- **RAM needed** — less than 200 MB
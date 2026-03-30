import streamlit as st

st.set_page_config(
    page_title="Text Analyzer Pro",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS 
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #333;
        margin-top: 1rem;
        color: #e0e0e0;
        line-height: 1.8;
    }
    .positive { color: #4ade80; font-size: 1.5rem; font-weight: 700; }
    .negative { color: #f87171; font-size: 1.5rem; font-weight: 700; }
    .stButton>button {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: black;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔍 Text Analyzer Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sentiment · Keywords · Statistics · Word Frequency</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "😊 Sentiment Analysis",
    "🔑 Keyword Extractor",
    "📊 Text Statistics",
    "🔤 Word Frequency"
])

# TAB 1: SENTIMENT 
with tab1:
    from sentiment import analyze_sentiment

    st.subheader("😊 Sentiment Analysis")
    st.markdown("Predicts whether your text is **Positive** or **Negative** using Logistic Regression.")

    samples = [
        "Select a sample...",
        "I absolutely loved this product. It works perfectly and the quality is amazing!",
        "This is the worst experience I have ever had. Completely disappointed.",
        "The movie was okay, nothing special but not bad either.",
        "Terrible service! I will never come back to this place again.",
        "Great quality and fast delivery. Highly recommended to everyone!",
    ]

    sample = st.selectbox("🎯 Try a sample:", samples, key="sent_sample")
    text_input = st.text_area(
        "Or type your text:",
        value="" if sample == samples[0] else sample,
        height=150,
        placeholder="Type any text to analyze sentiment...",
        key="sent_input"
    )

    if st.button("Analyze Sentiment →", key="sent_btn"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            result = analyze_sentiment(text_input)
            label      = result["label"]
            confidence = result["confidence"]
            css_class  = "positive" if label == "POSITIVE" else "negative"
            emoji      = "😊" if label == "POSITIVE" else "😞"

            st.markdown(f"""
            <div class="result-box">
                <span class="{css_class}">{emoji} {label}</span><br>
                <span style="color:#aaa;">Confidence: {confidence:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

            import pandas as pd
            scores_df = pd.DataFrame({
                "Sentiment": ["POSITIVE", "NEGATIVE"],
                "Score": [result["pos_score"], result["neg_score"]]
            }).set_index("Sentiment")
            st.bar_chart(scores_df)

# TAB 2: KEYWORD EXTRACTOR 
with tab2:
    from text_utils import extract_keywords

    st.subheader("🔑 Keyword Extractor")
    st.markdown("Extracts the most important keywords from your text using **TF-IDF**.")

    kw_text = st.text_area(
        "Paste your text here:",
        height=200,
        placeholder="Paste any article, paragraph or text...",
        key="kw_input"
    )
    top_n = st.slider("Number of keywords to extract:", 3, 20, 10)

    if st.button("Extract Keywords →", key="kw_btn"):
        if not kw_text.strip():
            st.warning("Please enter some text.")
        elif len(kw_text.split()) < 10:
            st.warning("Please enter at least 10 words.")
        else:
            keywords = extract_keywords(kw_text, top_n)
            import pandas as pd
            df = pd.DataFrame(keywords, columns=["Keyword", "Score"])
            df["Score"] = df["Score"].round(4)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Keywords:**")
                st.dataframe(df, use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**Importance Chart:**")
                st.bar_chart(df.set_index("Keyword"))

#  TAB 3: TEXT STATISTICS 
with tab3:
    from text_utils import get_text_stats

    st.subheader("📊 Text Statistics")
    st.markdown("Get detailed statistics about any text.")

    stats_text = st.text_area(
        "Paste your text here:",
        height=200,
        placeholder="Paste any text to get statistics...",
        key="stats_input"
    )

    if st.button("Get Statistics →", key="stats_btn"):
        if not stats_text.strip():
            st.warning("Please enter some text.")
        else:
            stats = get_text_stats(stats_text)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Words",      stats["total_words"])
            col2.metric("Unique Words",     stats["unique_words"])
            col3.metric("Total Sentences",  stats["sentences"])

            col4, col5, col6 = st.columns(3)
            col4.metric("Total Characters", stats["characters"])
            col5.metric("Avg Word Length",  stats["avg_word_length"])
            col6.metric("Avg Sentence Length", stats["avg_sentence_length"])

            st.markdown("**📋 Full Report:**")
            st.markdown(f"""
            <div class="result-box">
                📝 <b>Total Words:</b> {stats["total_words"]}<br>
                🔠 <b>Unique Words:</b> {stats["unique_words"]} ({stats["vocabulary_richness"]:.1%} vocabulary richness)<br>
                📖 <b>Sentences:</b> {stats["sentences"]}<br>
                🔤 <b>Characters (with spaces):</b> {stats["characters"]}<br>
                🔡 <b>Characters (no spaces):</b> {stats["characters_no_spaces"]}<br>
                📏 <b>Average Word Length:</b> {stats["avg_word_length"]} letters<br>
                📐 <b>Average Sentence Length:</b> {stats["avg_sentence_length"]} words<br>
                🔢 <b>Paragraphs:</b> {stats["paragraphs"]}<br>
            </div>
            """, unsafe_allow_html=True)

# TAB 4: WORD FREQUENCY 
with tab4:
    from text_utils import get_word_frequency

    st.subheader("🔤 Word Frequency")
    st.markdown("See which words appear most often in your text.")

    freq_text = st.text_area(
        "Paste your text here:",
        height=200,
        placeholder="Paste any text to see word frequency...",
        key="freq_input"
    )
    top_words = st.slider("Show top N words:", 5, 30, 15)

    if st.button("Analyze Frequency →", key="freq_btn"):
        if not freq_text.strip():
            st.warning("Please enter some text.")
        else:
            import pandas as pd
            freq_data = get_word_frequency(freq_text, top_words)
            df = pd.DataFrame(freq_data, columns=["Word", "Count"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Word Count Table:**")
                st.dataframe(df, use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**Frequency Chart:**")
                st.bar_chart(df.set_index("Word"))

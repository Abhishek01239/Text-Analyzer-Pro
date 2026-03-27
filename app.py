import streamlit as st

st.set_page_config(
    page_title = "Text Analyzer Pro",
    page_icon  = "🔍",
    layout = "wide"
)

st.markdown("""
<style>
            .main-title{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #11998e. ##38ef7d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            }
            .subtitle{
                color: #888;
            font-size: 1rem;
            margin-bottom: 2rem;
            }
            .result-box{
                background: #1e1e2e;
                border-radius: 12px;
                padding: 1.2rem 1.5rem;
                border: 1px solid #333;
                margin-top: 1rem;
                color: #e0e0e0;
                line-height: 1.8;
            }
            .positive{
                color: #4ade80;
                font-size: 1.5rem;
                font-weight: 700;    
            }
            .negative{
            color: #f87171;
            font-size: 1.5rem;
            font-weight: 700;
            }
            .stButton>button{
                background: linear-gradient(90 deg, #11998e, #38ef7d);
                color: black;
                border: none;
                border-radius: 8px;
                padding: 0.5rem 2rem;
                font-weight: 700;
            }
</style>
""", unsafe_allow_html = True)

st.amrkdown('<div class="main-title">🔍' )
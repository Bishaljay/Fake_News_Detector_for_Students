import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import re
import nltk
from datetime import datetime
import json

# ML imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="Student Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Download NLTK data
# -------------------------
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        with st.spinner(f"Downloading {resource}..."):
            nltk.download(resource)

# -------------------------
# Custom CSS (Enhanced UI)
# -------------------------
st.markdown("""
<style>
    body {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        font-size:3rem;
        color:#0077b6;
        text-align:center;
        margin-bottom:0.5rem;
        font-weight:bold;
    }
    .sub-header {
        text-align:center;
        font-size:1.2rem;
        color:#495057;
        margin-bottom:2rem;
    }
    .result-box {
        padding:20px;
        border-radius:15px;
        margin:15px 0;
        border:2px solid #dee2e6;
        background: white;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        text-align:center;
    }
    .fake {
        border-color:#dc3545; 
        background-color:#ffe5e5;
    }
    .reliable {
        border-color:#28a745; 
        background-color:#e7f9ee;
    }
    .borderline {
        border-color:#ffc107; 
        background-color:#fff9e6;
    }
    .footer {
        text-align:center;
        font-size:0.9rem;
        color:#868e96;
        margin-top:2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# FakeNewsDetector Class
# -------------------------
class FakeNewsDetector:
    def __init__(self):
        self.loaded = True

        # Small training dataset (expand as needed)
        train_texts = [
            "Breaking! Shocking miracle cure discovered",
            "Secret government conspiracy revealed",
            "Experts say climate change is real",
            "Research indicates vaccine is safe",
            "NASA confirms water on Mars",
            "Scandal leaked shocking fraud exposed",
            "According to university study data shows",
            "Peer-reviewed research confirms safety",
            "Government officially announced policy",
            "Fraud alert shocking news click here"
        ]
        train_labels = [
            "fake","fake","reliable","reliable","reliable",
            "fake","reliable","reliable","reliable","fake"
        ]

        self.vectorizer = CountVectorizer(stop_words="english")
        X_train = self.vectorizer.fit_transform(train_texts)
        self.clf = MultinomialNB()
        self.clf.fit(X_train, train_labels)

    def analyze_text(self, text):
        """Analyze text for fake news indicators"""
        if len(text) < 20:
            return {'error': 'Text too short for analysis'}
        
        summary = self.generate_summary(text)
        analysis = self.model_based_analysis(text)
        features = self.extract_features(text)
        
        return {
            'summary': summary,
            'analysis': analysis,
            'features': features,
            'word_count': len(text.split()),
            'char_count': len(text)
        }

    def model_based_analysis(self, text):
        """Offline ML-based Fake/Real detection"""
        try:
            X_test = self.vectorizer.transform([text])
            pred = self.clf.predict(X_test)[0]
            prob = self.clf.predict_proba(X_test)[0]

            if pred == "fake":
                verdict, color = "Fake News", "red"
                confidence = prob[self.clf.classes_.tolist().index("fake")]
            else:
                verdict, color = "Reliable", "green"
                confidence = prob[self.clf.classes_.tolist().index("reliable")]

            return {
                "verdict": verdict,
                "confidence": float(confidence),
                "color": color,
                "scores": {
                    "fake_score": float(prob[self.clf.classes_.tolist().index("fake")]),
                    "reliable_score": float(prob[self.clf.classes_.tolist().index("reliable")])
                }
            }
        except Exception as e:
            return {
                "verdict": "Error",
                "confidence": 0,
                "color": "orange",
                "scores": {"fake_score": 0, "reliable_score": 0},
                "error": str(e)
            }

    def extract_article_from_url(self, url):
        """Extract article content from URL"""
        try:
            headers = {'User-Agent':'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.find('title')
            title_text = title.get_text() if title else "No title found"
            
            content = ""
            article = soup.find('article')
            if article:
                content = article.get_text()
            else:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {'title': title_text, 'content': content, 'success': len(content) > 100}
        except Exception as e:
            return {'title': 'Error', 'content':'', 'success': False, 'error': str(e)}

    def generate_summary(self, text):
        """Extractive summary"""
        if not text or len(text) < 100: 
            return "Text too short for meaningful summary"
        try:
            sentences = nltk.sent_tokenize(text)
            return ' '.join(sentences[:3]) if len(sentences) > 3 else ' '.join(sentences)
        except Exception as e:
            return f"(Summary unavailable: {e})"

    def extract_features(self, text):
        words = nltk.word_tokenize(text.lower())
        sentences = nltk.sent_tokenize(text)
        sensational_words = ['shocking','miracle','secret','breaking','urgent','fraud','scandal']
        sensational_count = sum(1 for word in words if any(sw in word for sw in sensational_words))
        reliable_indicators = ['according','research','study','experts','official','confirmed']
        reliable_count = sum(1 for word in words if any(rw in word for rw in reliable_indicators))
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words)/len(sentences) if sentences else 0,
            'exclamation_count': text.count('!'),
            'sensational_word_count': sensational_count,
            'reliable_indicator_count': reliable_count
        }

# -------------------------
# Initialize detector in session
# -------------------------
if "detector" not in st.session_state:
    st.session_state.detector = FakeNewsDetector()
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# -------------------------
# Streamlit pages
# -------------------------
def render_home_page():
    st.markdown('<h1 class="main-header">📰 Fake News Detector for Students</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Empowering students to analyze, verify, and learn to spot misinformation.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://eecsnews.engin.umich.edu/wp-content/uploads/sites/2/2018/08/fake-news.jpg", width=450)
    with col2:
        st.markdown("""
        ### 🚀 Features
        - **🔍 Analyze** any article or pasted text  
        - **🎯 Assess** credibility with AI & keywords  
        - **📝 Summarize** long text into short, clear insights  
        - **📊 Track** history of your analyses  
        """)

def render_analysis_page():
    st.title("🔍 Analyze Article")
    input_method = st.radio("Choose input:", ["📝 Paste Text", "🌐 Enter URL"], horizontal=True)
    if input_method == "📝 Paste Text":
        input_content = st.text_area("Paste text here", height=200)
        analyze_btn = st.button("🚀 Analyze Text")
    else:
        url = st.text_input("Enter URL")
        analyze_btn = st.button("🌐 Analyze URL")
        input_content = url

    if analyze_btn and input_content:
        with st.spinner("Analyzing..."):
            if input_method == "🌐 Enter URL":
                result = st.session_state.detector.extract_article_from_url(input_content)
                if not result['success']:
                    st.error("Failed to extract content")
                    return
                article_title = result['title']
                article_content = result['content']
            else:
                article_title = "Pasted Text Analysis"
                article_content = input_content

            analysis_result = st.session_state.detector.analyze_text(article_content)
            if 'error' in analysis_result:
                st.error(analysis_result['error'])
                return
            full_result = {'timestamp': datetime.now().isoformat(), 'title': article_title, **analysis_result}
            st.session_state.analysis_history.append(full_result)
            display_results(full_result)

def display_results(result):
    st.success("✅ Analysis Complete!")
    
    col1,col2 = st.columns([2,1])
    
    with col1:
        st.subheader("📖 Article Info")
        st.write(f"**Title:** {result.get('title','N/A')}")
        st.write(f"**Words:** {result.get('word_count',0)}")
        st.subheader("📋 Summary")
        st.info(result['summary'])
    
    with col2:
        st.subheader("🌟 Believability")
        analysis = result['analysis']
        color_class = {"red":"fake","green":"reliable","orange":"borderline"}.get(analysis['color'],'borderline')
        
        st.markdown(f"""
        <div class="result-box {color_class}">
            <h2>{analysis['verdict']}</h2>
            <p style="font-size:1.1rem;">Confidence: <b>{analysis['confidence']:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("### 🧮 Score Breakdown")
        scores = analysis['scores']
        fig = px.pie(
            names=list(scores.keys()), 
            values=list(scores.values()), 
            color=list(scores.keys()), 
            color_discrete_map={'fake_score':'red','reliable_score':'green'},
            hole=0.4
        )
        fig.update_traces(textinfo='label+percent', pull=[0.05, 0.05])
        st.plotly_chart(fig, use_container_width=True)

def render_history_page():
    st.title("📈 Analysis History")
    if not st.session_state.analysis_history:
        st.info("No analyses yet")
        return
    for i, analysis in enumerate(st.session_state.analysis_history):
        with st.expander(f"Analysis {i+1}: {analysis.get('title','Unknown')}"):
            st.write(f"Verdict: {analysis['analysis']['verdict']}")
            st.write(f"Confidence: {st.session_state.analysis_history[i]['analysis']['confidence']:.1%}")
            st.write(f"Date: {pd.Timestamp(analysis['timestamp']).strftime('%Y-%m-%d %H:%M')}")

def render_learn_page():
    st.title("📖 Learn About Fake News")

    st.markdown("""
    ## 🛡️ How to Spot Fake News

    **🚨 Red Flags to Watch For:**  
    - Excessive capitalization or punctuation!!!  
    - Emotional or sensational language designed to provoke fear or anger.  
    - Vague claims or information with no supporting evidence.  
    - Unfamiliar or suspicious sources.  
    - Headlines that seem too shocking to be true.

    **✅ Signs of Reliable News:**  
    - Clear and credible sources cited.  
    - Multiple references or citations to verified information.  
    - Neutral, factual, and balanced language.  
    - Author and publication credentials are available.  
    - Supporting evidence such as images, studies, or links.

    **💡 Tips for Verifying News:**  
    - Cross-check information with multiple trusted sources.  
    - Verify images and videos using reverse image search tools.  
    - Check the publication date for timeliness.  
    - Question sensational claims and headlines.  
    - Use reliable fact-checking websites like Snopes or FactCheck.org.

    Stay alert, think critically, and don't share until you verify! 🧠
    """)


# -------------------------
# Run the app
# -------------------------
def main():
    with st.sidebar:
        st.title("📰 Student Fake News Detector")
        st.markdown("---")
        page = st.radio("Choose a page:", ["🏘️ Home", "📰 Analyze Article", "📚 History", "🎓 Learn"])
        st.markdown("---")
        st.metric("Analyses", len(st.session_state.analysis_history))
    
    if page == "🏘️ Home": render_home_page()
    elif page == "📰 Analyze Article": render_analysis_page()
    elif page == "📚  History": render_history_page()
    elif page == "🎓 Learn": render_learn_page()

    # Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        background-color: #ffffff;  /* white background */
        color: #1f51ff;             /* nice blue text color */
        font-size: 14px;
        padding: 8px 0;
        font-weight: bold;
        box-shadow: 0 -1px 3px rgba(0,0,0,0.1);
    }
    .footer:hover {
        color: #ff4b5c;             /* changes to red on hover */
    }
    </style>
    <p class="footer">© 2025 Bishal Jaysawal | Coding + ❤️ + Passion = Project</p>
    """,
    unsafe_allow_html=True
)

if __name__=="__main__":
    main()

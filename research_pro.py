import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import matplotlib as mpl
from wordcloud import WordCloud
from io import BytesIO
from docx import Document
import networkx as nx 
from langdetect import detect
import nltk

# --- 1. NLTK Resource Initialization ---
@st.cache_resource
def init_nltk():
    resources = ['cmudict', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'universal_tagset', 'punkt', 'punkt_tab']
    for res in resources:
        nltk.download(res, quiet=True)
    try:
        return nltk.corpus.cmudict.dict()
    except:
        return {}

cmu_dict = init_nltk()

# --- 2. Syllable Counting Functions ---
def count_syllables_en(word):
    word = word.lower()
    if word in cmu_dict:
        return max([len([list(y for y in x if y[-1].isdigit()) for x in cmu_dict[word]][0])])
    return len(re.findall(r'[aeiouy]+', word))

def count_syllables_th(word):
    try:
        from pythainlp.tokenize import syllable_tokenize
        return len(syllable_tokenize(word))
    except:
        return 0

# --- 3. Sentiment Analysis Function (Dual Language) ---
def analyze_sentiment(text, lang):
    if lang == 'th':
        pos = ['ดี', 'สำเร็จ', 'ภูมิใจ', 'ความสุข', 'พัฒนา', 'ประโยชน์', 'ยั่งยืน', 'พอเพียง', 'สะดวก']
        neg = ['ไม่ดี', 'ปัญหา', 'แย่', 'ยากลำบาก', 'ขาดแคลน', 'อุปสรรค', 'หนี้สิน', 'เดือดร้อน']
    else:
        pos = ['good', 'great', 'excellent', 'success', 'happy', 'positive', 'improve', 'benefit', 'sustainable']
        neg = ['bad', 'problem', 'difficult', 'lack', 'obstacle', 'debt', 'negative', 'poor', 'issue']
    
    low_text = text.lower()
    pos_score = sum(1 for w in pos if w in low_text)
    neg_score = sum(1 for w in neg if w in low_text)
    
    if pos_score > neg_score: return "Positive 😊"
    elif neg_score > pos_score: return "Negative 😟"
    else: return "Neutral 😐"

# --- 4. Font & Export Settings ---
font_path = "Kanit-Regular.ttf" 

def setup_font():
    try:
        mpl.font_manager.fontManager.addfont(font_path)
        prop = mpl.font_manager.FontProperties(fname=font_path)
        mpl.rc('font', family=prop.get_name(), size=12)
        mpl.rcParams['axes.unicode_minus'] = False 
        return prop
    except:
        return None

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis')
    return output.getvalue()

# --- 5. Main UI & Processing ---
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    THAI_READY = True
except:
    THAI_READY = False

st.set_page_config(layout="wide", page_title="Advanced Research Analysis Tool")
st.title("🔬 Advanced Research Analysis Tool")
st.subheader("Syllable Filter (5-10) & Sentiment Analysis")

if not THAI_READY:
    st.error("❌ Required libraries not found. Please check requirements.txt")
    st.stop()

font_p = setup_font()
uploaded_files = st.file_uploader("Upload Interview/Document Files (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    summary_for_all = []
    for file in uploaded_files:
        raw_text = file.read().decode("utf-8")
        try: lang = detect(raw_text)
        except: lang = 'th'
        
        # --- Filter Logic ---
        filtered_list = []
        if lang == 'th':
            tokens = word_tokenize(raw_text, keep_whitespace=False)
            stop_words = list(thai_stopwords())
            for t in tokens:
                t = t.strip()
                if t and not re.match(r'^[0-9\W]+$', t) and t not in stop_words:
                    if 5 <= count_syllables_th(t) <= 10:
                        filtered_list.append(t)
        else:
            words_only = re.findall(r'\b[a-zA-Z]{3,}\b', raw_text.lower())
            tagged = nltk.pos_tag(words_only)
            excluded = ['DT', 'CC', 'IN', 'PRP', 'PRP$', 'TO', 'MD', 'CD']
            for word, tag in tagged:
                if tag not in excluded:
                    if 5 <= count_syllables_en(word) <= 10:
                        filtered_list.append(word)

        word_counts = Counter(filtered_list)
        sentiment_result = analyze_sentiment(raw_text, lang)
        summary_for_all.append({
            "Filename": file.name, 
            "Language": "Thai" if lang=='th' else "English", 
            "Sentiment": sentiment_result
        })

        with st.expander(f"📊 Analysis Result: {file.name}", expanded=True):
            if not filtered_list:
                st.warning("🔍 No words found with 5-10 syllables.")
                st.write(f"**Sentiment Analysis:** {sentiment_result}")
            else:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("💡 Content Insights")
                    st.write(f"**Overall Sentiment:** {sentiment_result}")
                    
                    wc = WordCloud(width=800, height=450, background_color="white", font_path=font_path).generate(" ".join(filtered_list))
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
                    
                    buf_wc = BytesIO()
                    fig_wc.savefig(buf_wc, format="png")
                    st.download_button("💾 Download Word Cloud (PNG)", buf_wc.getvalue(), f"cloud_{file.name}.png", "image/png")

                with col2:
                    st.subheader("📈 Key Terms (5-10 Syllables)")
                    df_stats = pd.DataFrame(word_counts.most_common(20), columns=['Term', 'Frequency'])
                    st.table(df_stats)
                    st.download_button("🟢 Download Statistics (Excel)", to_excel(df_stats), f"stats_{file.name}.xlsx")
                
                # --- Network Analysis ---
                st.divider()
                st.subheader("🕸️ Semantic Network (High-Syllable Words)")
                G = nx.Graph()
                pairs = [tuple(sorted((filtered_list[i], filtered_list[i+1]))) for i in range(len(filtered_list)-1)]
                for p, w in Counter(pairs).most_common(15):
                    G.add_edge(p[0], p[1], weight=w)
                
                if len(G.nodes) > 0:
                    fig_net, ax_net = plt.subplots(figsize=(10, 6))
                    pos = nx.spring_layout(G, k=0.7)
                    nx.draw_networkx_edges(G, pos, edge_color='skyblue', alpha=0.4, width=2)
                    nx.draw_networkx_nodes(G, pos, node_color='salmon', node_size=1800)
                    for node, (x, y) in pos.items():
                        ax_net.text(x, y, node, fontproperties=font_p, fontsize=11, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    plt.axis('off')
                    st.pyplot(fig_net)
                    
                    buf_net = BytesIO()
                    fig_net.savefig(buf_net, format="png")
                    st.download_button("💾 Download Network (PNG)", buf_net.getvalue(), f"network_{file.name}.png", "image/png")
                else:
                    st.info("Insufficient data for network visualization.")

    # Global Summary Table
    st.divider()
    st.subheader("📋 Overall Project Summary")
    st.table(pd.DataFrame(summary_for_all))
else:
    st.info("👋 Welcome! Please upload .txt files to start the advanced analysis. Note: This tool filters specifically for academic or complex terms (5-10 syllables).")

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

# --- 1. ดาวน์โหลด NLTK Resources (แก้ไขปัญหา LookupError) ---
@st.cache_resource
def init_nltk():
    resources = [
        'cmudict', 
        'averaged_perceptron_tagger', 
        'averaged_perceptron_tagger_eng',
        'universal_tagset',
        'punkt',
        'punkt_tab'
    ]
    for res in resources:
        nltk.download(res, quiet=True)
    try:
        return nltk.corpus.cmudict.dict()
    except:
        return {}

cmu_dict = init_nltk()

# --- 2. ฟังก์ชันนับพยางค์ ---
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

# --- 3. ตั้งค่าฟอนต์และการแสดงผล ---
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

# --- 4. ฟังก์ชันหลักในการวิเคราะห์ ---
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    from pythainlp.summarize import summarize
    THAI_READY = True
except:
    THAI_READY = False

st.set_page_config(layout="wide", page_title="Ultimate Research Tool")
st.title("🔬 ระบบวิเคราะห์งานวิจัยขั้นสูง (Syllable Filter 5-10)")

if not THAI_READY:
    st.error("❌ Library พื้นฐานไม่พร้อมใช้งาน")
    st.stop()

font_p = setup_font()
uploaded_files = st.file_uploader("อัปโหลดไฟล์ (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        raw_text = file.read().decode("utf-8")
        try: lang = detect(raw_text)
        except: lang = 'th'
        
        # --- ระบบกรองคำ (Filter Logic) ---
        filtered_list = []
        if lang == 'th':
            tokens = word_tokenize(raw_text, keep_whitespace=False)
            stop_words = list(thai_stopwords()) + ['เนาะ', 'นะ', 'ครับ', 'ค่ะ', 'คือ', 'แบบ']
            for t in tokens:
                t = t.strip()
                if t and not re.match(r'^[0-9\W]+$', t) and t not in stop_words:
                    s_count = count_syllables_th(t)
                    if 5 <= s_count <= 10:
                        filtered_list.append(t)
        else:
            # ภาษาอังกฤษ: ใช้ POS Tagging กรอง Preposition, Article, etc.
            words_only = re.findall(r'\b[a-zA-Z]{3,}\b', raw_text.lower())
            tagged = nltk.pos_tag(words_only)
            # ตัดบทความ (DT), คำเชื่อม (CC), บุพบท (IN), คำสรรพนาม (PRP)
            excluded = ['DT', 'CC', 'IN', 'PRP', 'PRP$', 'TO', 'MD', 'CD']
            for word, tag in tagged:
                if tag not in excluded:
                    s_count = count_syllables_en(word)
                    if 5 <= s_count <= 10:
                        filtered_list.append(word)

        word_counts = Counter(filtered_list)
        
        with st.expander(f"📊 ผลการวิเคราะห์: {file.name} ({'ไทย' if lang=='th' else 'EN'})", expanded=True):
            if not filtered_list:
                st.warning("🔍 ไม่พบคำที่มีความยาว 5-10 พยางค์ตามเงื่อนไข")
            else:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("☁️ Word Cloud (5-10 Syllables)")
                    wc = WordCloud(width=800, height=450, background_color="white", font_path=font_path).generate(" ".join(filtered_list))
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                    
                    buf_wc = BytesIO()
                    fig.savefig(buf_wc, format="png")
                    st.download_button("💾 ดาวน์โหลดรูปภาพ (PNG)", buf_wc.getvalue(), f"cloud_{file.name}.png", "image/png")

                with col2:
                    st.subheader("📈 สถิติคำสำคัญ")
                    df_stats = pd.DataFrame(word_counts.most_common(20), columns=['คำ', 'จำนวนครั้ง'])
                    st.table(df_stats)
                    st.download_button("🟢 ดาวน์โหลดข้อมูล (Excel)", to_excel(df_stats), f"stats_{file.name}.xlsx")
                
                # --- Network Analysis สำหรับคำกรอง ---
                st.divider()
                st.subheader("🕸️ โครงข่ายความสัมพันธ์ของคำยาว")
                G = nx.Graph()
                pairs = [tuple(sorted((filtered_list[i], filtered_list[i+1]))) for i in range(len(filtered_list)-1)]
                for p, w in Counter(pairs).most_common(15):
                    G.add_edge(p[0], p[1], weight=w)
                
                if len(G.nodes) > 0:
                    fig_net, ax_net = plt.subplots(figsize=(10, 6))
                    pos = nx.spring_layout(G, k=0.6)
                    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
                    nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=1500)
                    for node, (x, y) in pos.items():
                        ax_net.text(x, y, node, fontproperties=font_p, fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    plt.axis('off')
                    st.pyplot(fig_net)
                else:
                    st.info("ข้อมูลไม่พอสำหรับสร้างโครงข่าย")

else:
    st.info("💡 คำแนะนำ: ระบบจะแสดงเฉพาะคำวิชาการหรือคำเฉพาะที่มีความยาว **5-10 พยางค์** เท่านั้น (เช่น 'กตัญญูกตเวที' หรือ 'Sustainability')")

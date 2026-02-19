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
from nltk.corpus import cmudict # สำหรับนับพยางค์ภาษาอังกฤษ

# ดาวน์โหลดข้อมูลที่จำเป็นสำหรับ NLTK
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')

d = cmudict.dict()

# --- ฟังก์ชันนับพยางค์ภาษาอังกฤษ ---
def count_syllables_en(word):
    word = word.lower()
    if word in d:
        return max([len([list(y for y in x if y[-1].isdigit()) for x in d[word]][0])])
    # กรณีไม่พบในดิกชันนารี ใช้การนับสระพื้นฐาน
    return len(re.findall(r'[aeiouy]+', word))

# --- ฟังก์ชันนับพยางค์ภาษาไทย ---
def count_syllables_th(word):
    try:
        from pythainlp.tokenize import syllable_tokenize
        return len(syllable_tokenize(word))
    except:
        return 0

# --- 1. ตั้งค่าการแสดงผลภาษาไทย ---
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

# --- 2. ฟังก์ชันช่วยสร้างไฟล์ Export ---
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Summary')
    return output.getvalue()

# --- 3. ฟังก์ชันวิเคราะห์อารมณ์ ---
def analyze_sentiment(text, lang):
    if lang == 'th':
        pos = ['ดี', 'สำเร็จ', 'ภูมิใจ', 'ความสุข', 'พัฒนา', 'ประโยชน์', 'ยั่งยืน', 'พอเพียง']
        neg = ['ไม่ดี', 'ปัญหา', 'แย่', 'ยากลำบาก', 'ขาดแคลน', 'อุปสรรค', 'หนี้สิน']
    else:
        pos = ['good', 'great', 'excellent', 'success', 'happy', 'positive', 'improve', 'benefit']
        neg = ['bad', 'problem', 'difficult', 'lack', 'obstacle', 'debt', 'negative', 'poor']
    
    pos_score = sum(1 for w in pos if w in text.lower())
    neg_score = sum(1 for w in neg if w in text.lower())
    return "บวก 😊" if pos_score > neg_score else ("ลบ 😟" if neg_score > pos_score else "ปกติ 😐")

# --- เริ่มต้นโปรแกรม ---
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    from pythainlp.summarize import summarize
    THAI_READY = True
except:
    THAI_READY = False

st.set_page_config(layout="wide", page_title="Syllable Filter Research Tool")
st.title("🕸️ ระบบวิเคราะห์งานวิจัย (กรองคำ 5-10 พยางค์)")

if not THAI_READY:
    st.error("❌ Library ไม่พร้อมใช้งาน")
    st.stop()

font_p = setup_font()
uploaded_files = st.file_uploader("อัปโหลดไฟล์ (.txt)", type=['txt'], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        raw_text = file.read().decode("utf-8")
        try: lang = detect(raw_text)
        except: lang = 'th'
        
        # --- ระบบกรองคำตามเงื่อนไขใหม่ ---
        if lang == 'th':
            tokens = word_tokenize(raw_text, keep_whitespace=False)
            stop_words = list(thai_stopwords())
            filtered = []
            for t in tokens:
                t = t.strip()
                # ตัดสัญลักษณ์/ตัวเลข และคำที่สั้นเกินไป
                if t and not re.match(r'^[0-9\W]+$', t) and t not in stop_words:
                    syl_count = count_syllables_th(t)
                    if 5 <= syl_count <= 10: # กรองพยางค์ 5-10
                        filtered.append(t)
        else:
            # ภาษาอังกฤษ: ตัด Preposition, Article, Conjunction, Number
            import nltk
            words_only = re.findall(r'\b[a-zA-Z]+\b', raw_text.lower())
            tagged = nltk.pos_tag(words_only)
            # ตัดบทความ (DT), คำเชื่อม (CC), บุพบท (IN)
            excluded_tags = ['DT', 'CC', 'IN', 'PRP', 'PRP$', 'TO', 'MD']
            filtered = []
            for word, tag in tagged:
                if tag not in excluded_tags and len(word) > 2:
                    syl_count = count_syllables_en(word)
                    if 5 <= syl_count <= 10: # กรองพยางค์ 5-10
                        filtered.append(word)

        word_counts = Counter(filtered)
        filtered_final = [w for w in filtered if word_counts[w] >= 1] # ปรับให้โชว์แม้เจอแค่ครั้งเดียว

        with st.expander(f"📑 ผลการวิเคราะห์: {file.name} ({'ไทย' if lang=='th' else 'EN'})", expanded=True):
            if not filtered_final:
                st.warning("⚠️ ไม่พบคำที่มีความยาว 5-10 พยางค์ในไฟล์นี้")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**จำนวนคำที่ผ่านการกรอง:** {len(filtered_final)} คำ")
                    wc = WordCloud(width=800, height=400, background_color="white", font_path=font_path).generate(" ".join(filtered_final))
                    fig, ax = plt.subplots()
                    ax.imshow(wc)
                    ax.axis("off")
                    st.pyplot(fig)
                with c2:
                    st.subheader("📈 สถิติคำ (5-10 พยางค์)")
                    df_counts = pd.DataFrame(word_counts.most_common(15), columns=['คำ', 'จำนวนครั้ง'])
                    st.table(df_counts)
                    st.download_button("🟢 โหลด Excel", to_excel(df_counts), f"filter_{file.name}.xlsx")

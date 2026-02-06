# 1. ติดตั้ง Library ที่จำเป็น
!pip install streamlit pyngrok feedparser transformers plotly -q

# 2. เขียนโค้ดลงไฟล์ app.py (เราจะไม่รันตรงๆ แต่จะบันทึกเป็นไฟล์ก่อน)
%%writefile app.py

import streamlit as st
import feedparser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import urllib.parse
from dateutil import parser
import time

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Thai News Sentiment Dashboard", layout="wide")

st.title("📰 Real-time Thai News Sentiment Dashboard")
st.markdown("วิเคราะห์อารมณ์ข่าวเศรษฐกิจและการเมืองไทย ด้วย AI & Hybrid Filtering")

# --- ส่วน Sidebar (เมนูซ้ายมือ) ---
st.sidebar.header("⚙️ ตั้งค่าการดึงข้อมูล")
keyword = st.sidebar.text_input("คำค้นหา (Keyword)", "เศรษฐกิจไทย")
num_news = st.sidebar.slider("จำนวนข่าวที่ต้องการ", 10, 50, 20)
auto_refresh = st.sidebar.checkbox("Auto Refresh (ทุก 5 นาที)")

# --- Load AI Model (Cache ไว้ จะได้ไม่โหลดใหม่ทุกครั้งที่กด) ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

try:
    with st.spinner('กำลังโหลด AI Model...'):
        sentiment_analyzer = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- Keyword Filters ---
negative_keywords = ["หนี้", "ร่วง", "ทรุด", "วิกฤต", "เสี่ยง", "กังวล", "ต่ำ", "ลดลง", "กระทบ", "ปัญหา", "แพง", "ซบเซา", "เจ๊ง", "แดงเดือด", "ระวัง"]
positive_keywords = ["พุ่ง", "โต", "ฟื้น", "สดใส", "สำเร็จ", "ยอดเยี่ยม", "กำไร", "บวก", "เชื่อมั่น", "ดีขึ้น", "เฮ", "อนุมัติ", "New High", "แจกเงิน"]

def analyze_hybrid(text):
    # Rule-based
    for word in negative_keywords:
        if word in text: return "Negative", -0.9
    for word in positive_keywords:
        if word in text: return "Positive", 0.9
    # AI-based
    result = sentiment_analyzer(text[:512])[0]
    score = result['score']
    if result['label'] == 'positive': return "Positive", score
    if result['label'] == 'negative': return "Negative", -score
    return "Neutral", 0

# --- Function ดึงข่าว ---
def get_news_data(key, limit):
    encoded_keyword = urllib.parse.quote(key)
    rss_url = f"https://news.google.com/rss/search?q={encoded_keyword}&hl=th&gl=TH&ceid=TH:th"
    feed = feedparser.parse(rss_url)
    
    data = []
    for entry in feed.entries[:limit]:
        title = entry.title
        pub_date = parser.parse(entry.published)
        sentiment, score = analyze_hybrid(title)
        data.append({
            "Title": title,
            "Sentiment": sentiment,
            "Score": score,
            "Date": pub_date,
            "Link": entry.link
        })
    return pd.DataFrame(data)

# --- Main Interface ---
if st.sidebar.button("🔄 อัปเดตข้อมูลเดี๋ยวนี้") or auto_refresh:
    with st.spinner(f'กำลังดึงข่าว "{keyword}"...'):
        df = get_news_data(keyword, num_news)
        
        if not df.empty:
            # 1. Key Metrics (ตัวเลขสรุปด้านบน)
            col1, col2, col3, col4 = st.columns(4)
            total = len(df)
            pos = len(df[df['Sentiment']=='Positive'])
            neg = len(df[df['Sentiment']=='Negative'])
            neu = len(df[df['Sentiment']=='Neutral'])
            
            col1.metric("ข่าวทั้งหมด", total)
            col2.metric("ข่าวดี (Positive)", pos, delta_color="normal")
            col3.metric("ข่าวร้าย (Negative)", neg, delta_color="inverse") # สีแดงถ้าเยอะ
            col4.metric("ทั่วไป (Neutral)", neu)

            # 2. Charts Layout
            c1, c2 = st.columns([1, 2]) # แบ่งหน้าจอ ซ้าย 1 ส่วน ขวา 2 ส่วน
            
            with c1:
                # Interactive Pie Chart (Plotly)
                fig_pie = px.pie(df, names='Sentiment', title='สัดส่วนอารมณ์ข่าว',
                                 color='Sentiment',
                                 color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#FECB52'})
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                # Interactive Scatter/Trend Chart
                df = df.sort_values(by='Date')
                fig_line = px.scatter(df, x='Date', y='Score', color='Sentiment',
                                      hover_data=['Title'], 
                                      title='แนวโน้มอารมณ์ข่าวตามเวลา (Timeline)',
                                      color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#FECB52'})
                fig_line.update_traces(marker=dict(size=12))
                fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_line, use_container_width=True)

            # 3. Data Table
            st.subheader("📋 รายการข่าวล่าสุด")
            # แสดงเฉพาะคอลัมน์ที่จำเป็น และทำให้ลิงก์กดได้ (ต้องใช้ท่า HTML นิดหน่อยถ้าจะทำ Link จริงจัง แต่เบื้องต้นโชว์ Text ไปก่อน)
            st.dataframe(df[['Date', 'Sentiment', 'Title', 'Score']], use_container_width=True)

        else:
            st.warning("ไม่พบข้อมูลข่าว")
            
else:
    st.info("👈 กดปุ่ม 'อัปเดตข้อมูลเดี๋ยวนี้' ที่เมนูด้านซ้าย เพื่อเริ่มการทำงาน")

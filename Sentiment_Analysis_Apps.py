import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import emoji
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
import nltk
from collections import Counter
import seaborn as sns
import plotly.express as px


nltk_data_dir = "./resources/nltk_data_dir/"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("punkt", download_dir=nltk_data_dir)

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

def extract_ngrams(text, n=2):
    tokens = text.split()
    return list(nltk.ngrams(tokens, n))

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = tokenizer.tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    return ''

@st.cache_data
def preprocess_text_column(data):
    data['sentiment'] = data['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')
    data['review'] = data['review'].fillna('').apply(preprocess_text)
    data['review_length'] = data['review'].apply(lambda x: len(x.split()))
    return data

def train_model(data):
    start_time = time.time()
    new_data = preprocess_text_column(data)
    X_train, X_test, y_train, y_test = train_test_split(new_data['review'], new_data['sentiment'], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    end_time = time.time()

    st.session_state['new_data'] = new_data
    st.session_state['vectorizer'] = vectorizer
    st.session_state['y_test'] = y_test
    st.session_state['y_pred'] = y_pred

    execution_time = end_time-start_time
    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True), confusion_matrix(y_test, y_pred), execution_time


st.title("EDA Sentiment Analysis")

with st.sidebar:
    st.header(":material/note_add: New Project")
    st.info("Select a dataset and model you want to train.", icon=":material/info:")
    filename = st.file_uploader("Upload your data:", type=["csv", "xlsx"])
    with st.form("model_form", border=False):
        model_name = st.selectbox("Select the model:", ["Naive Bayes (NB)"])
        submitted = st.form_submit_button("Run", icon=":material/model_training:", type="primary")

with st.container():
    st.header("Data Preview", divider="gray")
    if filename:
        data = pd.read_csv(filename) if filename.name.endswith('.csv') else pd.read_excel(filename)
        st.session_state.data = data
        st.write("Preview of uploaded data:")
        st.dataframe(data.head(10), hide_index=True, use_container_width=True)
    else:
        st.info("Please upload a dataset.")
    
with st.container():
    st.header("📊 Model Output", divider="gray")
    if submitted and 'data' in st.session_state:
        with st.spinner("Training model..."):
            accuracy, report, cm, execution_time = train_model(st.session_state.data)
            st.success("✅ Model training complete!")
            st.metric("Accuracy", f"{accuracy:.2%}")
            st.write(f"⏱️ Training Time: {execution_time:.2f} seconds")
            with st.expander("📝 Classification Report"):
                c1, c2 = st.columns([1, 0.8], border=False)
                with c1:
                    st.subheader(":material/summarize: Report")
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                    st.caption("This classification report provides an analysis of the performance of a machine learning model in predicting classes or categories")
                with c2:
                    st.subheader("📊 Precision, Recall, F1-Score")
                    st.caption("This chart displays the performance of a classification model across three key metrics")
                    st.bar_chart(report_df[['precision', 'recall', 'f1-score']].drop('accuracy', errors="ignore"))
                st.subheader("📉 Confusion Matrix")
                st.caption("The confusion matrix displays the predicted results of a classification model compared to the actual labels. This matrix helps identify the strengths and weaknesses of the model in predicting classes.")
                labels = ['Negative', 'Positive']
                fig_cm = px.imshow(cm, text_auto=True, aspect='auto', color_continuous_scale='Blues',
                       x=labels, y=labels, title="Confusion Matrix")
                st.plotly_chart(fig_cm)
                
    elif submitted:
        st.error("❌ Please upload a dataset first!")
    else:
        st.info("Click 'Run' in the sidebar to initiate the training process, which will utilize the selected machine learning model to optimize the dataset for accurate predictions.")

with st.container():
    st.header("🔍 Data Insights")
    if "new_data" in st.session_state:
        new_data = st.session_state['new_data']
        st.subheader("📑 Preprocessed Dataset")
        st.dataframe(new_data.head(10), hide_index=True, use_container_width=True)
        
        col1, col2 = st.columns([2,1], gap='small', vertical_alignment="center")
        col3, col4 = st.columns([1,1])
        col5, col6 = st.columns([2,1], vertical_alignment='center')
        col7, col8 = st.columns([1,0.8], vertical_alignment="center")
        
        with col1:
            st.subheader("📏 Review Length Distribution")
            fig_length = px.histogram(new_data, x='review_length',
                                      nbins=50, title="Distribution of Review Text Length",
                                      labels={"Review_length" : "text Length (Words)"},
                                      color_discrete_sequence=["blue"])
            st.plotly_chart(fig_length)
        
        with col2:
            #st.subheader("Data Statistic", divider='gray')
            st.caption("Overview of the review length in dataset, including key statistics such as count, mean, and distribution of numerical values.")
            st.write(new_data['review_length'].describe())
            
        with col3:
            sentiment_counts = new_data['sentiment'].value_counts()
            fig_sentiment = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
                           labels={'x': 'Sentiment', 'y': 'Count'},
                           title='Sentiment Distribution', color=sentiment_counts.index,
                           color_discrete_sequence=['green', 'red'])
            st.plotly_chart(fig_sentiment)
        
        with col4:
            fig_boxplt = px.box(new_data, x='sentiment', y='review_length', color="sentiment",
                                color_discrete_sequence=["green", "red"])
            st.plotly_chart(fig_boxplt)

        with col5:
            st.subheader("📌 Top 20 Most Frequent Words")
            all_words = ' '.join(new_data["review"]).split()
            common_words = Counter(all_words).most_common(20)
            words, counts = zip(*common_words)

            fig_words = px.bar(x=counts, y=words, orientation='h', 
                                title="Top 20 Most Frequent Words",
                                labels={'x': 'Frequency', 'y': 'Words'},
                                color_discrete_sequence=['blue'])
            fig_words.update_yaxes(categoryorder='total ascending')
            
            st.plotly_chart(fig_words)

            
        with col6:
            word_freq_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
            st.dataframe(word_freq_df, hide_index=True)

        with col7:
            st.subheader("📌 Top 10 Most Frequent Bigrams")
            all_bigrams = []
            for review in new_data['review']:
                all_bigrams.extend(extract_ngrams(review, n=2))
            
            bigram_freq = Counter(all_bigrams)
            top_10_bigrams = bigram_freq.most_common(10)
            #top_10_bigrams_df = pd.DataFrame(top_10_bigrams, columns=['Bigram', 'Frekuensi'])
            fig_bigrams = px.bar(top_10_bigrams, y=[str(bigram[0]) for bigram in top_10_bigrams], 
                                 x=[bigram[1] for bigram in top_10_bigrams], orientation='h',
                                 title="Top 10 Most Frequent Bigrams",
                                 labels={"x" : "Frequency",
                                         "y" : "Bigrams"},
                                 color_discrete_sequence=["blue"])
            fig_bigrams.update_yaxes(categoryorder='total ascending')
            
            st.plotly_chart(fig_bigrams)
        
        with col8:
            bigram_freq_df = pd.DataFrame(top_10_bigrams, columns=["Bigram", "Frequency"])
            st.dataframe(bigram_freq_df, hide_index=True)


        with st.container():
            st.subheader("☁️ Word Cloud")
            wordcloud = WordCloud(width=600, height=300, background_color='white').generate(' '.join(st.session_state.new_data['review']))
    
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
    else:
        st.info("Data visualization will appear here.")

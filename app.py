import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import gc

# Page configuration
st.set_page_config(
    page_title="Text Analysis App",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'nlp' not in st.session_state:
    try:
        with st.spinner("Loading spaCy model..."):
            st.session_state.nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        st.stop()

if 'sentiment_pipeline' not in st.session_state:
    try:
        with st.spinner("Loading sentiment analysis model..."):
            st.session_state.sentiment_pipeline = pipeline("sentiment-analysis")
    except Exception as e:
        st.error(f"Failed to load sentiment analysis model: {str(e)}")
        st.stop()

# Main title
st.title("📊 Text Analysis Dashboard")
st.markdown("Analyze comments and text data with various NLP techniques")

# Sidebar for input
st.sidebar.header("Input Data")

# Text input options
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Upload CSV", "Paste Text", "Sample Data"]
)

comments = []

if input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'comments' in df.columns or 'text' in df.columns:
                column = 'comments' if 'comments' in df.columns else 'text'
                comments = df[column].dropna().tolist()
                
                # Limit to reasonable number for performance
                if len(comments) > 1000:
                    st.sidebar.warning(f"Large dataset detected ({len(comments)} comments). For better performance, consider using a smaller sample.")
                    if st.sidebar.button("Use first 1000 comments"):
                        comments = comments[:1000]
                        st.sidebar.success(f"Using first {len(comments)} comments")
                    else:
                        st.sidebar.info("Processing all comments may take longer...")
                else:
                    st.sidebar.success(f"Loaded {len(comments)} comments")
            else:
                st.sidebar.error("CSV must contain 'comments' or 'text' column")
                st.sidebar.info("Available columns: " + ", ".join(df.columns.tolist()))
        except Exception as e:
            st.sidebar.error(f"Error reading CSV file: {str(e)}")

elif input_method == "Paste Text":
    text_input = st.sidebar.text_area("Enter comments (one per line):", height=200)
    if text_input:
        comments = [line.strip() for line in text_input.split('\n') if line.strip()]
        st.sidebar.success(f"Loaded {len(comments)} comments")

else:  # Sample Data
    sample_comments = [
        "This product is amazing! I love it so much.",
        "Terrible quality, would not recommend to anyone.",
        "The service was okay, nothing special.",
        "Outstanding customer support, very helpful team.",
        "Product arrived damaged, very disappointed.",
        "Great value for money, will buy again.",
        "Poor packaging, item was broken.",
        "Excellent quality and fast shipping!",
        "Not what I expected, but it's usable.",
        "Perfect! Exactly what I was looking for."
    ]
    comments = sample_comments
    st.sidebar.success(f"Using {len(comments)} sample comments")

# Main content area
if comments:
    st.header("Analysis Results")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Sentiment Analysis", 
        "📝 Summarization", 
        "🏷️ Topic Modeling", 
        "☁️ Word Cloud", 
        "🔍 Named Entity Recognition"
    ])
    
    with tab1:
        st.subheader("Sentiment Analysis")
        if st.button("Analyze Sentiment", key="sentiment_btn"):
            with st.spinner("Analyzing sentiment..."):
                try:
                    results = st.session_state.sentiment_pipeline(comments)
                    
                    # Create results DataFrame
                    df_results = pd.DataFrame({
                        'Comment': comments,
                        'Sentiment': [r['label'] for r in results],
                        'Confidence': [r['score'] for r in results]
                    })
                    
                    st.dataframe(df_results)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
                        st.metric("Positive", positive_count)
                    with col2:
                        negative_count = sum(1 for r in results if r['label'] == 'NEGATIVE')
                        st.metric("Negative", negative_count)
                    with col3:
                        avg_confidence = np.mean([r['score'] for r in results])
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                        
                except Exception as e:
                    st.error(f"Error in sentiment analysis: {str(e)}")
    
    with tab2:
        st.subheader("Text Summarization")
        if st.button("Generate Summaries", key="summarize_btn"):
            with st.spinner("Generating summaries..."):
                try:
                    summarizer = pipeline("summarization")
                    summaries = []
                    
                    for i, comment in enumerate(comments):
                        if len(comment.split()) > 50:  # Only summarize longer texts
                            try:
                                summary = summarizer(
                                    comment, 
                                    max_length=130, 
                                    min_length=30, 
                                    do_sample=False
                                )
                                summaries.append({
                                    'Original': comment,
                                    'Summary': summary[0]["summary_text"]
                                })
                            except Exception as e:
                                summaries.append({
                                    'Original': comment,
                                    'Summary': f"Error summarizing: {str(e)}"
                                })
                        else:
                            summaries.append({
                                'Original': comment,
                                'Summary': "Text too short for summarization"
                            })
                    
                    for i, summary in enumerate(summaries):
                        with st.expander(f"Comment {i+1}"):
                            st.write("**Original:**", summary['Original'])
                            st.write("**Summary:**", summary['Summary'])
                            
                except Exception as e:
                    st.error(f"Error in summarization: {str(e)}")
                    st.info("This might be due to network issues or model loading problems. Please try again.")
    
    with tab3:
        st.subheader("Topic Modeling")
        if st.button("Perform Topic Modeling", key="topics_btn"):
            with st.spinner("Performing topic modeling..."):
                try:
                    if len(comments) < 5:
                        st.warning("Need at least 5 comments for topic modeling")
                    else:
                        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                        tfidf = vectorizer.fit_transform(comments)
                        kmeans = KMeans(n_clusters=min(5, len(comments)), random_state=42)
                        kmeans.fit(tfidf)
                        labels = kmeans.labels_
                        
                        # Group comments by topic
                        topics = {}
                        for i, label in enumerate(labels):
                            if label not in topics:
                                topics[label] = []
                            topics[label].append(comments[i])
                        
                        # Display topics
                        for topic_id, topic_comments in topics.items():
                            with st.expander(f"Topic {topic_id + 1} ({len(topic_comments)} comments)"):
                                for comment in topic_comments:
                                    st.write(f"• {comment}")
                                    
                except Exception as e:
                    st.error(f"Error in topic modeling: {str(e)}")
    
    with tab4:
        st.subheader("Word Cloud")
        if st.button("Generate Word Cloud", key="wordcloud_btn"):
            with st.spinner("Generating word cloud..."):
                try:
                    text = " ".join(comments)
                    wc = WordCloud(width=1000, height=500, background_color='white').generate(text)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Word Cloud of Comments', fontsize=16)
                    
                    st.pyplot(fig)
                    
                    # Download option
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight')
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    st.download_button(
                        label="Download Word Cloud",
                        data=buffer.getvalue(),
                        file_name="wordcloud.png",
                        mime="image/png"
                    )
                    
                    # Clean up memory
                    plt.close(fig)
                    buffer.close()
                    gc.collect()
                    
                except Exception as e:
                    st.error(f"Error generating word cloud: {str(e)}")
                    st.info("This might be due to insufficient memory or text processing issues.")
    
    with tab5:
        st.subheader("Named Entity Recognition")
        if st.button("Extract Entities", key="ner_btn"):
            with st.spinner("Extracting entities..."):
                try:
                    all_entities = []
                    for comment in comments:
                        doc = st.session_state.nlp(comment)
                        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
                        all_entities.append({
                            'Comment': comment,
                            'Entities': entities
                        })
                    
                    # Display results
                    for i, result in enumerate(all_entities):
                        if result['Entities']:
                            with st.expander(f"Comment {i+1}"):
                                st.write("**Text:**", result['Comment'])
                                st.write("**Entities:**")
                                for entity in result['Entities']:
                                    st.write(f"• {entity['text']} ({entity['label']})")
                        else:
                            with st.expander(f"Comment {i+1} (No entities found)"):
                                st.write(result['Comment'])
                                
                except Exception as e:
                    st.error(f"Error in NER: {str(e)}")

else:
    st.info("👈 Please provide some text data using the sidebar to get started!")
    
    # Show sample data preview
    st.subheader("Sample Data Preview")
    sample_data = [
        "This product is amazing! I love it so much.",
        "Terrible quality, would not recommend to anyone.",
        "The service was okay, nothing special.",
        "Outstanding customer support, very helpful team.",
        "Product arrived damaged, very disappointed."
    ]
    
    st.write("Here's what the sample data looks like:")
    for i, comment in enumerate(sample_data, 1):
        st.write(f"{i}. {comment}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Powered by Transformers and spaCy")

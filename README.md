# Text Analysis Streamlit App

A comprehensive text analysis dashboard built with Streamlit that provides sentiment analysis, text summarization, topic modeling, word cloud generation, and named entity recognition.

## Features

- **Sentiment Analysis**: Analyze the sentiment of text comments using Hugging Face transformers
- **Text Summarization**: Generate summaries of longer text content
- **Topic Modeling**: Group similar comments into topics using K-means clustering
- **Word Cloud**: Visualize the most frequent words in your text data
- **Named Entity Recognition**: Extract named entities (people, places, organizations) from text

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 3. Run the App

```bash
streamlit run app.py
```

## Usage

1. **Input Data**: Choose from three input methods:
   - Upload a CSV file with 'comments' or 'text' column
   - Paste text directly (one comment per line)
   - Use sample data for testing

2. **Analysis**: Click the buttons in each tab to perform different analyses:
   - Sentiment Analysis: Get sentiment labels and confidence scores
   - Summarization: Generate summaries for longer texts
   - Topic Modeling: Group comments into topics
   - Word Cloud: Visualize word frequency
   - NER: Extract named entities

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with the following settings:
   - Main file path: `app.py`
   - Python version: 3.9+

### Local Deployment

```bash
streamlit run app.py --server.port 8501
```

## Requirements

- Python 3.8+
- 4GB+ RAM (for model loading)
- Internet connection (for downloading models)

## Troubleshooting

- **Model Loading Issues**: Ensure you have enough RAM and a stable internet connection
- **spaCy Model Error**: Run `python -m spacy download en_core_web_sm`
- **Memory Issues**: Try reducing the number of comments or restart the app

## File Structure

```
streamlit_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

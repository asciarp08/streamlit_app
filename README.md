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

### Docker (recommended for sharing)

First, install Docker Desktop: [Get Docker](https://docs.docker.com/get-docker/)

#### Build image
```bash
docker build -t text-analysis-app:latest .
```

#### Run container
```bash
# Map host 8501 to container 8501
docker run --rm -p 8501:8501 --name text-analysis-app text-analysis-app:latest

# If 8501 is busy, use another host port (e.g., 8502)
docker run --rm -p 8502:8501 --name text-analysis-app text-analysis-app:latest
```
Then open `http://localhost:8501` (or the host port you chose).

#### Share the image
- Push to a registry (Docker Hub example):
```bash
docker tag text-analysis-app:latest YOUR_DOCKERHUB_USERNAME/text-analysis-app:latest
docker push YOUR_DOCKERHUB_USERNAME/text-analysis-app:latest
```
Recipient runs:
```bash
docker pull YOUR_DOCKERHUB_USERNAME/text-analysis-app:latest
docker run --rm -p 8501:8501 --name text-analysis-app YOUR_DOCKERHUB_USERNAME/text-analysis-app:latest
```

- Offline tarball:
```bash
docker save -o text-analysis-app.tar text-analysis-app:latest
# recipient
docker load -i text-analysis-app.tar
docker run --rm -p 8501:8501 --name text-analysis-app text-analysis-app:latest
```

#### Multi-arch build (optional)
For sharing across Apple Silicon (arm64) and x86 (amd64):
```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t YOUR_DOCKERHUB_USERNAME/text-analysis-app:latest --push .
```

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
├── Dockerfile          # Container build instructions
├── .dockerignore       # Docker build context ignore rules
└── README.md           # Project documentation
```

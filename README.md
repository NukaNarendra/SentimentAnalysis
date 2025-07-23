# 📊 Social Media Sentiment Analysis

## Project Overview
This comprehensive project analyzes sentiment and topics from social media text data using advanced Natural Language Processing (NLP), machine learning, and deep learning techniques. By processing Twitter data through multiple analytical layers, we extract valuable insights about public opinion, trending topics, and emotional responses to various subjects. The resulting dashboard visualizes these insights through interactive charts and word clouds, making complex data patterns easily understandable.

## 📁 Project Structure

```
Social-Media-Sentiment-Analysis/
├── preprocess.py       # Text cleaning, stopword removal, language detection
├── analysis.py         # Sentiment scoring, KMeans clustering, entity extraction, topic modeling
├── app.py              # Streamlit dashboard for visualization
├── requirements.txt    # Project dependencies
└── README.md           # Project overview and usage instructions
```

## 🧪 Features & Methodology

### Data Preprocessing (`preprocess.py`)
- **Text Cleaning**: Removes URLs, emojis, mentions, hashtags, and special characters
- **Normalization**: Converts text to lowercase and removes extra whitespace
- **Stopword Removal**: Filters out common English stopwords using NLTK
- **Language Detection**: Uses the `langdetect` library to filter for English tweets only
- **Data Filtering**: Handles missing values and removes duplicate entries

```python
# Example of the text cleaning function
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r"@\w+|#\w+", "", text)  # Remove mentions & hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Keep only letters and spaces
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text
```

### Sentiment & Topic Analysis (`analysis.py`)
- **Sentiment Analysis**: 
  - Uses `TextBlob` to calculate polarity (positive/negative) and subjectivity scores
  - Polarity ranges from -1 (negative) to 1 (positive)
  - Subjectivity ranges from 0 (objective) to 1 (subjective)
  
- **Sentiment Clustering**:
  - Applies K-means clustering to group tweets with similar sentiment patterns
  - Standardizes features to ensure equal weighting of polarity and subjectivity
  - Maps clusters to sentiment scores (1-5) based on average polarity
  
- **Named Entity Recognition**:
  - Uses `spaCy`'s pre-trained models to identify organizations, people, products, locations, and brands
  - Calculates frequency of entities to identify key players in discussions
  - Filters entities by type to focus on relevant categories

- **Topic Modeling**:
  - Generates sentence embeddings using `sentence-transformers` (all-mpnet-base-v2)
  - Applies K-means clustering to group semantically similar content
  - Identifies dominant words in each cluster to label topics
  - Creates vector representations that capture deeper semantic meaning beyond simple word frequency

```python
# Example of sentiment extraction
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity
```

### Interactive Dashboard (`app.py`)
- **Framework**: Built with `Streamlit` for rapid web app deployment
- **Visualization Libraries**: Utilizes `Plotly` for interactive charts and `WordCloud` for visual text analysis
- **Key Visualizations**:
  - Sentiment distribution bar chart showing emotion patterns across dataset
  - Interactive scatter plot of polarity vs. subjectivity with hover data
  - Dynamic word clouds for each sentiment cluster highlighting key terms
  - Bar chart of top named entities showing key discussion subjects
  - Topic cluster overview with representative keywords

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Social-Media-Sentiment-Analysis.git
cd Social-Media-Sentiment-Analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
python -m spacy download en_core_web_sm
```

### 3. Prepare Your Data
The input dataset should be a CSV file with a text column (default column name is `clean_text`). You can modify the column name in the code if needed.

### 4. Run the Pipeline
```bash
# Step 1: Clean and prepare the data
python preprocess.py

# Step 2: Perform sentiment analysis and clustering
python analysis.py

# Step 3: Launch the dashboard
streamlit run app.py
```

## 📊 Dashboard Components

### Sentiment Distribution
Visualizes the distribution of sentiment scores (1-5) across the dataset, providing an immediate overview of public opinion (positive, negative, or neutral).

### Polarity vs. Subjectivity
An interactive scatter plot that maps tweets based on their sentiment polarity (x-axis) and subjectivity (y-axis). Points are colored by cluster, and hovering reveals the actual tweet text.

### Word Clouds
Generates visual representations of the most frequent words in each sentiment cluster, with word size proportional to frequency. This provides an intuitive understanding of the vocabulary associated with different emotional responses.

### Top Named Entities
Displays the most frequently mentioned organizations, people, products, locations, and brands in the dataset, highlighting key subjects of discussion.

### Topic Clusters
Shows the dominant words for each topic cluster, helping to identify the main themes and subjects being discussed across the dataset.

## 💻 Technical Implementation Details

### Embedding Generation
The project uses the `all-mpnet-base-v2` model from the `sentence-transformers` library to generate high-quality vector representations of tweets. These 768-dimensional embeddings capture semantic relationships between texts far better than traditional bag-of-words approaches.

### Clustering Algorithm
We use K-means clustering with 5 clusters (configurable) for both sentiment and topic clustering. The number of clusters can be adjusted based on dataset size and desired granularity.

```python
# Example of the clustering implementation
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(features)
```

### Performance Considerations
- The embedding and entity recognition processes are computationally intensive for large datasets
- For datasets >10,000 tweets, consider sampling or batch processing
- The Streamlit dashboard is optimized for datasets up to 100,000 tweets

## 🔧 Customization Options

### Adjusting Cluster Count
```python
# In analysis.py, modify the n_clusters parameter
kmeans = KMeans(n_clusters=YOUR_DESIRED_NUMBER, random_state=42, n_init=10)
```

### Changing Embedding Model
```python
# In analysis.py, replace the model name with an alternative from sentence-transformers
model = SentenceTransformer('YOUR_PREFERRED_MODEL')
```

### Adding New Visualizations
The Streamlit framework makes it easy to extend the dashboard with additional visualizations. See the Streamlit documentation for more options.

## 📌 Best Practices & Notes

- **Data Quality**: Better preprocessing leads to more accurate sentiment analysis
- **Language Considerations**: The model performs best on English text; use language filtering
- **Entity Recognition**: May need tuning for domain-specific terms
- **Dashboard Performance**: Limit the number of tweets displayed in interactive elements for better performance
- **Sentiment Interpretation**: Consider industry benchmarks when interpreting sentiment scores

## 🔮 Future Enhancements

- **Temporal Analysis**: Adding time-based tracking of sentiment changes
- **Comparative Analysis**: Facilitating comparison between different topics or time periods
- **Advanced NLP**: Integration with more sophisticated models like BERT or GPT
- **Real-time Processing**: Adding capability for stream processing of incoming tweets
- **Geospatial Analysis**: Mapping sentiment patterns by geographic location

## 📬 Contact & Contribution

For suggestions, issues, or contributions, please:
- Open an issue on GitHub
- Submit a pull request with proposed changes
- Contact the maintainer at: `nukanarendra2006@gmail.com`

---

*This project was developed as part of an effort to better understand public opinion through social media analysis using modern NLP techniques.*

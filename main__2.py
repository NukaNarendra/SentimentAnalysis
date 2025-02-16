import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet

# Load spaCy model (only once)
nlp = spacy.load('en_core_web_sm')

# Load dataset
file_path = 'twitter sentiment analysis after cleaning.csv'
df = pd.read_csv(file_path)

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Error detecting language for text: {text[:30]}... - {str(e)}")
        return 'unknown'

# Function to get sentiment score
def get_sentiment(text, lang):
    if lang == 'en':
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    return None, None

# Function to extract key points
def extract_key_points(text, lang):
    if lang == 'en':
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    return []

# Apply language detection, sentiment analysis, and key point extraction
df['language'] = df['Text'].apply(detect_language)
df['polarity'], df['subjectivity'] = zip(*df.apply(lambda row: get_sentiment(row['Text'], row['language']), axis=1))
df['key_points'] = df.apply(lambda row: extract_key_points(row['Text'], row['language']), axis=1)

# Save processed data
df.to_csv('processed_sentiment_data.csv', index=False)

# Load processed data
df = pd.read_csv('processed_sentiment_data.csv')

# Fill missing values
df[['polarity', 'subjectivity']] = df[['polarity', 'subjectivity']].fillna(df[['polarity', 'subjectivity']].median())

# Standardizing the data
X = df[['polarity', 'subjectivity']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualization: Clustering
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis', marker='o')
plt.title('KMeans Clustering of Sentiment Data')
plt.xlabel('Polarity (Standardized)')
plt.ylabel('Subjectivity (Standardized)')
plt.colorbar(label='Cluster Label')
plt.show()

# Regression Analysis: Linear Regression
X_reg = df[['polarity']]
y_reg = df['subjectivity']
regr = LinearRegression()
regr.fit(X_reg, y_reg)
df['predicted_subjectivity'] = regr.predict(X_reg)

# Visualization: Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(df['polarity'], df['subjectivity'], color='blue', label='Actual')
plt.plot(df['polarity'], df['predicted_subjectivity'], color='red', linewidth=2, label='Predicted')
plt.title('Linear Regression: Polarity vs Subjectivity')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.legend()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['polarity', 'subjectivity']].corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap of Polarity and Subjectivity')
plt.show()

# ElasticNet Regression
regr_en = ElasticNet(fit_intercept=True)
regr_en.fit(X_reg, y_reg)
df['predicted_subjectivity_en'] = regr_en.predict(X_reg)

# Visualization: ElasticNet Regression
plt.figure(figsize=(8, 6))
plt.scatter(df['polarity'], df['subjectivity'], color='blue', label='Actual')
plt.plot(df['polarity'], df['predicted_subjectivity_en'], color='green', linewidth=2, label='ElasticNet Predicted')
plt.title('ElasticNet Regression: Polarity vs Subjectivity')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.legend()
plt.show()


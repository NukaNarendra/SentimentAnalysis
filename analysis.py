from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

df['polarity'], df['subjectivity'] = zip(*df['cleaned_text'].apply(get_sentiment))

# Normalize
scaler = StandardScaler()
features = scaler.fit_transform(df[['polarity', 'subjectivity']])

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(features)

# Assign sentiment score to each cluster
# (You can sort by average polarity per cluster for better mapping)
cluster_avg = df.groupby('cluster')['polarity'].mean().sort_values()
sentiment_mapping = {cluster: i+1 for i, cluster in enumerate(cluster_avg.index)}

df['sentiment_score'] = df['cluster'].map(sentiment_mapping)


import spacy

nlp = spacy.load("en_core_web_sm")
key_entities = {}

for text in df['cleaned_text']:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "PRODUCT", "GPE", "BRAND"]:
            ent_text = ent.text.lower()
            key_entities[ent_text] = key_entities.get(ent_text, 0) + 1

# Top 10
from collections import Counter
top_entities = dict(Counter(key_entities).most_common(10))


!pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

# Generate embeddings
embeddings = model.encode(df['cleaned_text'].tolist())

# Cluster embeddings
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['topic_cluster'] = kmeans.fit_predict(embeddings)

# Show top words per cluster
from collections import Counter

def get_dominant_words(texts, num_words=5):
    words = []
    for text in texts:
        words.extend(text.split())
    return [word for word, _ in Counter(words).most_common(num_words)]

for cluster_id in range(5):
    texts = df[df['topic_cluster'] == cluster_id]['cleaned_text'].tolist()
    print(f"Cluster {cluster_id}: {get_dominant_words(texts)}")

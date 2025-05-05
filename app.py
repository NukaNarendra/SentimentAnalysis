# Save the following as app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

# Load cleaned dataset
df = pd.read_csv("final_output.csv")

st.title("Twitter Sentiment Analysis Dashboard")

# Sentiment Distribution
st.subheader("Sentiment Distribution")
sentiment_counts = df['sentiment_score'].value_counts().sort_index()
fig1 = px.bar(x=sentiment_counts.index, y=sentiment_counts.values, labels={"x": "Sentiment Score", "y": "Count"})
st.plotly_chart(fig1)

# Polarity vs Subjectivity
st.subheader("Polarity vs Subjectivity")
fig2 = px.scatter(df, x='polarity', y='subjectivity', color='cluster', hover_data=['cleaned_text'])
st.plotly_chart(fig2)

# Word Clouds per Sentiment Cluster
st.subheader("Word Clouds per Sentiment Cluster")
for cluster_id in df['cluster'].unique():
    cluster_text = ' '.join(df[df['cluster'] == cluster_id]['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    st.write(f"Cluster {cluster_id}")
    st.image(wordcloud.to_array())

# Top Named Entities
st.subheader("Top Named Entities")
top_entities = {"entity1": 15, "entity2": 12, "entity3": 10}  # Replace with real data
if top_entities:
    fig3 = px.bar(x=list(top_entities.keys()), y=list(top_entities.values()), labels={"x": "Entity", "y":"Frequency"})
    st.plotly_chart(fig3)

# Topic Clusters Overview
st.subheader("Topic Clusters (Top Words)")
for i in range(5):
    st.write(f"Cluster {i}: Add your keywords or labels here...")

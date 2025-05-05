import pandas as pd

# Load dataset
df = pd.read_csv("/content/Twitter_Data.csv")

# Remove unwanted column if exists
if 'category' in df.columns:
    df.drop('category', axis=1, inplace=True)

# Ensure text column is named consistently
df.rename(columns={df.columns[0]: "clean_text"}, inplace=True)

import re
import nltk
import emoji
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r"@\w+|#\w+", "", text)  # Remove mentions & hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Keep only letters and spaces
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['cleaned_text'] = df['clean_text'].apply(clean_text)

from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return None

df['language'] = df['cleaned_text'].apply(detect_language)
df = df[df['language'] == 'en']
df.drop('language', axis=1, inplace=True)

# Optional: Save for reuse
df.to_csv('filtered_data.csv', index=False)

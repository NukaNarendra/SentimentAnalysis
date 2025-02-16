# Social Media Sentiment Analysis

## 📌 Project Overview
This project aims to analyze Twitter sentiment data to understand brand perception using **Natural Language Processing (NLP) and machine learning techniques**. The project involves sentiment classification, clustering, and regression analysis to uncover trends in social media discussions.

## 🎯 Objectives
- **Sentiment Analysis**: Categorize tweets as **Positive, Negative, or Neutral** using `spaCy`, `TextBlob`, and `LangDetect`.
- **Clustering**: Group tweets based on sentiment scores using **K-Means clustering**.
- **Trend Analysis**: Examine sentiment trends over time to understand audience reactions.
- **Regression Analysis**: Use **Linear Regression & ElasticNet** to explore relationships between sentiment polarity and subjectivity.

## 📂 Dataset Description
- **Columns:**
  - `Text` – The tweet content.
  - `Label` – Sentiment classification (Positive, Negative, Neutral).
  - `Polarity` – Sentiment polarity score (-1 to 1).
  - `Subjectivity` – Sentiment subjectivity score (0 to 1).
- **Processed Data Output**: `processed_sentiment_data.csv` (includes computed sentiment scores and clusters).

## 🛠 Tools & Technologies Used
- **Python**: Primary language for analysis.
- **Pandas & NumPy**: Data preprocessing and manipulation.
- **NLP Libraries**: `spaCy`, `TextBlob`, `LangDetect` for text processing.
- **Machine Learning**:
  - `K-Means` for sentiment clustering.
  - `Linear Regression` & `ElasticNet` for sentiment prediction.
- **Visualization**: `Matplotlib` & `Seaborn` for data visualization.

## 📊 Methodology
1. **Data Preprocessing**
   - Remove URLs, special characters, and stopwords.
   - Perform **language detection** and filter non-English text.
   - Apply **sentiment analysis** to determine polarity and subjectivity.
   
2. **Clustering & Trend Analysis**
   - Standardize polarity & subjectivity scores.
   - Apply **K-Means clustering** to group tweets based on sentiment trends.
   
3. **Regression Analysis**
   - Perform **Linear Regression** and **ElasticNet Regression** to analyze relationships between polarity and subjectivity.
   - Generate visualizations to interpret patterns.

## 🚀 Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```
2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn spacy textblob langdetect scikit-learn
   ```
3. **Download the `spaCy` language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. **Run the sentiment analysis script:**
   ```bash
   python sentiment_analysis.py
   ```

## 📈 Expected Outcomes
✔ **Sentiment classification** (Positive, Negative, Neutral) from Twitter data.
✔ **Visualizations** showing sentiment distribution and trends.
✔ **Insights into brand perception** through sentiment clustering.
✔ **Predictive modeling** to analyze relationships between polarity and subjectivity.

## 🔮 Future Enhancements
- **Integrate Twitter API** to fetch real-time tweets.
- **Analyze engagement metrics** (likes, retweets, comments) to improve insights.
- **Use Topic Modeling (LDA)** to extract key discussion themes.
- **Incorporate sales data** to analyze sentiment-sales correlation.

## 💡 Conclusion
This project provides deep insights into **brand sentiment trends on Twitter**, helping businesses understand audience reactions. While it does not include direct sales analysis, it lays a strong foundation for **social media impact assessment**.

---
Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.



📩 **For queries, reach out via email or GitHub Issues.** 🚀


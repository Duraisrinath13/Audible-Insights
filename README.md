# 🎧 Audible Insights: Intelligent Book Recommendations

A book recommendation system built on Audible catalog data. It cleans and merges two datasets, uses **NLP (TF-IDF)** and **K-Means clustering** to understand books, and recommends them through **three models** in a **Streamlit** web app.

**Domain:** Recommendation Systems
**Skills:** Python, Data Cleaning, EDA, NLP, Machine Learning, Clustering, Streamlit, AWS

---

## 📌 Problem Statement

Design a book recommendation system that processes and cleans book data, applies NLP and clustering, builds multiple recommendation models, and lets users find books through a user-friendly Streamlit app.

## 💼 Business Use Cases

- **Personalized reading:** recommend books based on a book the user liked, their interests or favourite genres
- **Libraries and bookstores:** suggest popular and similar books
- **Authors and publishers:** insights about popular genres and reader preferences
- **Reader engagement:** trending books and hidden gems

---

## 📂 Project Structure

```
├── app.py                          # Streamlit app
├── requirements.txt
├── .streamlit/config.toml          # App theme
├── data/
│   ├── Audible_Catlog.csv                       # Dataset 1 (raw)
│   ├── Audible_Catlog_Advanced_Features.csv     # Dataset 2 (raw)
│   ├── cleaned_books.csv                        # Output of notebook 01
│   └── books_with_clusters.csv                  # Output of notebook 03 (used by app)
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.npz
│   └── kmeans_model.pkl
└── notebooks/
    ├── 01_data_cleaning.ipynb
    ├── 02_eda.ipynb
    └── 03_nlp_clustering_models.ipynb
```

---

## 🔄 Approach

### 1. Data Cleaning and Merging (`01_data_cleaning.ipynb`)
| Issue found | How it was handled |
|---|---|
| Same book in both files | Merged on Book Name + Author (outer join, normalized to lowercase) |
| Duplicate books | Kept the most complete record |
| Rating = -1 (not rated) | Changed to NaN (no fake ratings) |
| Missing review counts | Filled with 0 |
| ~1,800 junk descriptions (web-scraping errors such as captcha pages) | Detected and removed |
| Listening Time as text ("8 hours and 19 minutes") | Converted to minutes |
| Ranks and Genre in one string | Split into genre list, primary genre and overall rank |

**Result:** 6,063 unique books.

### 2. Exploratory Data Analysis (`02_eda.ipynb`)
- **Most popular genres:** Personal Success, Self-Help, Business
- **Highest-rated authors (min 3 books):** Tui T. Sutherland, Alan Watts, Mary Pope Osborne
- **Ratings:** mostly 4.0–5.0 (average 4.46), left skewed
- **Ratings vs reviews:** books with few reviews have unreliable ratings; with more reviews, ratings settle around 4.5
- **Publication year:** not present in the dataset

### 3. NLP and Clustering (`03_nlp_clustering_models.ipynb`)
- **TF-IDF** on title + genres + description (5,000 features, English stop words removed)
- **K-Means** with k = 30, chosen using the elbow method and silhouette score
- **DBSCAN** tested, but it marked too many books as noise, so K-Means was chosen

### 4. Recommendation Models
| Model | How it works |
|---|---|
| Content-Based | Cosine similarity between TF-IDF vectors |
| Cluster-Based | Most popular books from the same K-Means cluster |
| **Hybrid** | 70% text similarity + 20% popularity + 10% same cluster |

Popularity score = rating × log(1 + number of reviews), scaled to 0–1.

### 5. Evaluation
The data has no user-level ratings, so a recommendation counts as **relevant if it shares a genre** with the selected book. Tested on 300 books.

| Model | Precision@10 | Recall@10 | RMSE |
|---|---|---|---|
| Content-Based | 0.392 | 0.120 | 0.347 |
| Cluster-Based | 0.194 | 0.027 | 0.362 |
| **Hybrid** | **0.434** | 0.109 | 0.347 |

✅ **Hybrid is the best model** and is the default in the app.

### 6. Scenario Questions
- **New user who likes Science Fiction:** The Martian, Ready Player One, Mockingjay, 1984, The Testaments
- **User who rated thrillers highly:** Jack Reacher, Dan Brown, Chetan Bhagat, Rebecca
- **Hidden gems:** 217 books rated ≥ 4.7 with only 5–50 reviews

---

## 🖥️ Streamlit App

| Page | Feature |
|---|---|
| 🏠 Home | Catalog stats and trending books |
| 📖 Similar Books | Pick a book and choose Hybrid, Content or Cluster model |
| 💡 Search by Interest | Type a topic and get matching books |
| 🎭 Browse by Genre | Best books in your favourite genres |
| 💎 Hidden Gems | Highly rated books with few reviews |
| 📊 Insights | Interactive EDA charts and model results |

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/Duraisrinath13/Audible-Insights.git
cd Audible-Insights
pip install -r requirements.txt
streamlit run app.py
```

To rebuild the data and models, run the notebooks in order: 01 → 02 → 03.

## ☁️ Deployment

Deployed on **AWS EC2** (Ubuntu, t3.small), with Streamlit running on port 8501.

---

## 🛠️ Tech Stack
Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit · AWS EC2

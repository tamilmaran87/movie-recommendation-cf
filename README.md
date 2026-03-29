# 🎬 Movie Recommendation System

A collaborative filtering-based recommendation engine built on the **MovieLens 100K dataset**.  
Supports both **User-Based** and **Item-Based** filtering using cosine similarity.

---

## 📌 Features

- **User-Based CF** — Recommends movies by finding users with similar taste
- **Item-Based CF** — Recommends movies similar to a given movie
- **Cosine Similarity** — Used to compute user-user and item-item similarity matrices
- **Weighted Prediction** — Predicts ratings using weighted average of similar users
- Auto-downloads MovieLens 100K dataset if not already present

---

## 🛠️ Tech Stack

| Category        | Tools / Libraries                     |
|----------------|---------------------------------------|
| Language        | Python                                |
| Data Processing | Pandas, NumPy                         |
| ML / Similarity | Scikit-learn (cosine_similarity)      |
| Dataset         | MovieLens 100K (GroupLens Research)   |
| Environment     | Jupyter Notebook / Google Colab       |

---

## 📂 Project Structure

```
movie-recommendation/
│
├── data/                        # Auto-downloaded MovieLens 100K dataset
│   └── ml-100k/
│       ├── u.data               # User ratings (user_id, movie_id, rating, timestamp)
│       └── u.item               # Movie metadata (movie_id, title, ...)
│
├── src/
│   └── recommender.py           # Core CollaborativeFilteringRecommender class
│
└── README.md
```

---

## ⚙️ How It Works

### 1. Data Loading
- Downloads and extracts **MovieLens 100K** automatically
- Loads ratings and movie metadata into Pandas DataFrames

### 2. User-Movie Matrix
- Builds a pivot table: rows = users, columns = movies, values = ratings
- Missing ratings filled with 0

### 3. Similarity Computation
- **User-User similarity** → cosine similarity on the user-movie matrix
- **Item-Item similarity** → cosine similarity on the transposed matrix

### 4. Recommendations
- **User-Based:** Finds top 20 similar users → computes weighted average ratings for unseen movies → returns top-N
- **Item-Based:** Finds top-N movies most similar to a given movie

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas scikit-learn
```

### Usage

```python
from src.recommender import CollaborativeFilteringRecommender

# Initialize and train
rec = CollaborativeFilteringRecommender()
rec.fit()

# Get movie recommendations for User ID 1
recommendations = rec.recommend_for_user(user_id=1, top_n=10)
for movie in recommendations:
    print(movie['title'], "-", movie['score'])

# Get similar movies to Movie ID 50
similar = rec.recommend_similar_movies(movie_id=50, top_n=10)
for movie in similar:
    print(movie['title'], "-", movie['score'])
```

---

## 📊 Dataset

- **MovieLens 100K** by GroupLens Research, University of Minnesota
- 100,000 ratings from 943 users on 1,682 movies
- Auto-downloaded from: https://files.grouplens.org/datasets/movielens/ml-100k.zip

---

## 🧠 Concepts Used

- Collaborative Filtering (User-Based & Item-Based)
- Cosine Similarity
- Weighted Average Rating Prediction
- Pivot Table / Matrix Operations
- Data Preprocessing with Pandas

---

## 👨‍💻 Author

**TamilMaran R**  
B.Tech – Artificial Intelligence & Data Science  
Vel Tech University, Chennai  
GitHub: [github.com/tamilmaran87](https://github.com/tamilmaran87)

"""
Collaborative Filtering Recommender Engine
Uses User-Based and Item-Based CF on MovieLens 100K dataset
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def download_movielens():
    """Download and extract MovieLens 100K if not already present."""
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")
    extracted = os.path.join(DATA_DIR, "ml-100k")

    if not os.path.exists(extracted):
        os.makedirs(DATA_DIR, exist_ok=True)
        print("[INFO] Downloading MovieLens 100K …")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        os.remove(zip_path)
        print("[INFO] Download complete.")
    else:
        print("[INFO] MovieLens 100K already present.")


def load_data():
    """Load ratings and movies into DataFrames."""
    download_movielens()
    ml_dir = os.path.join(DATA_DIR, "ml-100k")

    ratings = pd.read_csv(
        os.path.join(ml_dir, "u.data"),
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    movies = pd.read_csv(
        os.path.join(ml_dir, "u.item"),
        sep="|",
        encoding="latin-1",
        usecols=[0, 1],
        names=["movie_id", "title"],
    )

    return ratings, movies


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CollaborativeFilteringRecommender:
    """User-Based and Item-Based Collaborative Filtering."""

    def __init__(self):
        self.ratings = None
        self.movies = None
        self.user_movie_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.is_fitted = False

    # ---- Fit ---------------------------------------------------------------

    def fit(self):
        """Load data and compute similarity matrices."""
        self.ratings, self.movies = load_data()

        # Build user-movie matrix
        self.user_movie_matrix = self.ratings.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        ).fillna(0)

        # User-User similarity
        print("[INFO] Computing user similarity …")
        self.user_similarity = cosine_similarity(self.user_movie_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index,
        )

        # Item-Item similarity
        print("[INFO] Computing item similarity …")
        self.item_similarity = cosine_similarity(self.user_movie_matrix.T)
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
            index=self.user_movie_matrix.columns,
            columns=self.user_movie_matrix.columns,
        )

        self.is_fitted = True
        print("[INFO] Model ready.")
        return self

    # ---- User-Based CF -----------------------------------------------------

    def recommend_for_user(self, user_id: int, top_n: int = 10):
        """Return top-N movie recommendations for a given user (User-Based CF)."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        if user_id not in self.user_movie_matrix.index:
            raise ValueError(f"User {user_id} not found in dataset.")

        # Similar users (excluding self)
        sim_scores = self.user_similarity_df[user_id].drop(user_id).sort_values(ascending=False)
        top_users = sim_scores.head(20).index

        # Movies not yet rated by the target user
        rated_by_user = set(
            self.ratings[self.ratings["user_id"] == user_id]["movie_id"]
        )

        # Weighted average of ratings from similar users
        weighted_ratings = {}
        similarity_sums = {}

        for similar_user in top_users:
            weight = sim_scores[similar_user]
            user_ratings = self.user_movie_matrix.loc[similar_user]

            for movie_id, rating in user_ratings.items():
                if movie_id not in rated_by_user and rating > 0:
                    weighted_ratings[movie_id] = weighted_ratings.get(movie_id, 0) + weight * rating
                    similarity_sums[movie_id] = similarity_sums.get(movie_id, 0) + weight

        predicted = {
            mid: weighted_ratings[mid] / similarity_sums[mid]
            for mid in weighted_ratings
            if similarity_sums[mid] > 0
        }

        top_movies = sorted(predicted.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return self._format_recommendations(top_movies)

    # ---- Item-Based CF -----------------------------------------------------

    def recommend_similar_movies(self, movie_id: int, top_n: int = 10):
        """Return movies similar to a given movie (Item-Based CF)."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        if movie_id not in self.item_similarity_df.index:
            raise ValueError(f"Movie {movie_id} not found in dataset.")

        sim_scores = (
            self.item_similarity_df[movie_id]
            .drop(movie_id)
            .sort_values(ascending=False)
            .head(top_n)
        )

        results = [(mid, score) for mid, score in sim_scores.items()]
        return self._format_recommendations(results)

    # ---- Helpers -----------------------------------------------------------

    def _format_recommendations(self, movie_score_list):
        """Attach movie titles to (movie_id, score) pairs."""
        records = []
        for movie_id, score in movie_score_list:
            title_row = self.movies[self.movies["movie_id"] == movie_id]
            title = title_row["title"].values[0] if not title_row.empty else "Unknown"
            records.append({
                "movie_id": int(movie_id),
                "title": title,
                "score": round(float(score), 4),
            })
        return records

    def get_all_movies(self):
        """Return full movie list."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return self.movies.to_dict(orient="records")

    def get_all_users(self):
        """Return sorted list of user IDs."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return sorted(self.user_movie_matrix.index.tolist())

    def get_user_rated_movies(self, user_id: int):
        """Return movies already rated by a user."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        user_ratings = self.ratings[self.ratings["user_id"] == user_id].merge(
            self.movies, on="movie_id"
        )[["movie_id", "title", "rating"]].sort_values("rating", ascending=False)
        return user_ratings.to_dict(orient="records")
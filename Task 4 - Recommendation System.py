"""
Task 4: Recommendation System
CodSoft AI Internship
Author: Doodakula Mohammad Abdulla

Implements both:
  • Collaborative Filtering  — user-user similarity (cosine)
  • Content-Based Filtering  — item-feature TF-IDF similarity

Dataset: MovieLens-style synthetic data (no download required)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ─── Synthetic Dataset ────────────────────────────────────────────────────────

MOVIES = pd.DataFrame({
    "movie_id": range(1, 16),
    "title": [
        "Inception", "The Dark Knight", "Interstellar", "The Matrix",
        "Avengers: Endgame", "Parasite", "The Godfather", "Pulp Fiction",
        "Forrest Gump", "The Shawshank Redemption",
        "Toy Story", "Finding Nemo", "Up", "WALL-E", "Coco"
    ],
    "genres": [
        "Sci-Fi Thriller",   "Action Thriller",   "Sci-Fi Drama",     "Sci-Fi Action",
        "Action Superhero",  "Drama Thriller",     "Crime Drama",      "Crime Drama",
        "Drama Romance",     "Drama",
        "Animation Family",  "Animation Family",   "Animation Drama",  "Animation Sci-Fi",
        "Animation Family"
    ],
    "director": [
        "Nolan", "Nolan", "Nolan", "Wachowski",
        "Russo", "Bong",  "Coppola", "Tarantino",
        "Zemeckis", "Darabont",
        "Lasseter", "Stanton", "Docter", "Stanton", "Molina"
    ],
    "year": [
        2010, 2008, 2014, 1999,
        2019, 2019, 1972, 1994,
        1994, 1994,
        1995, 2003, 2009, 2008, 2017
    ]
})

# Ratings matrix: users × movies (0 = not rated)
RATINGS_DATA = {
    "Alice":   [5, 4, 5, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Bob":     [4, 5, 4, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Charlie": [0, 0, 0, 0, 0, 5, 5, 4, 5, 5, 0, 0, 0, 0, 0],
    "Diana":   [0, 0, 0, 0, 0, 4, 4, 5, 4, 5, 0, 0, 0, 0, 0],
    "Eve":     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 4, 5, 4],
    "Frank":   [3, 0, 3, 0, 0, 0, 0, 0, 3, 4, 4, 5, 5, 4, 5],
    "Grace":   [5, 5, 0, 4, 5, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    "Heidi":   [0, 0, 0, 0, 0, 5, 0, 4, 5, 4, 3, 4, 0, 0, 5],
}

RATINGS = pd.DataFrame(RATINGS_DATA, index=MOVIES["title"]).T


# ─── Collaborative Filtering (User-User) ──────────────────────────────────────

class CollaborativeFilteringRecommender:
    """
    User-User Collaborative Filtering.
    Finds users with similar taste and recommends what they liked.
    """

    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings
        # Replace 0 with NaN for mean-centering
        data = ratings.replace(0, np.nan)
        # Mean-center each user's ratings
        self.centered = data.subtract(data.mean(axis=1), axis=0).fillna(0)
        self.similarity = pd.DataFrame(
            cosine_similarity(self.centered),
            index=ratings.index,
            columns=ratings.index
        )

    def recommend(self, user: str, top_n: int = 5) -> pd.Series:
        if user not in self.ratings.index:
            raise ValueError(f"User '{user}' not found.")

        sim_scores = self.similarity[user].drop(user)
        rated_by_user = self.ratings.loc[user]
        unrated_movies = rated_by_user[rated_by_user == 0].index

        predicted = {}
        for movie in unrated_movies:
            # Weighted average of other users' ratings
            others = self.ratings[movie]
            mask = others > 0
            if mask.sum() == 0:
                continue
            weighted = (sim_scores[mask] * others[mask]).sum()
            sim_total = sim_scores[mask].abs().sum()
            if sim_total > 0:
                predicted[movie] = weighted / sim_total

        return (pd.Series(predicted)
                  .sort_values(ascending=False)
                  .head(top_n))


# ─── Content-Based Filtering ──────────────────────────────────────────────────

class ContentBasedRecommender:
    """
    Content-Based Filtering using TF-IDF on movie metadata.
    Recommends movies similar to ones the user has already rated highly.
    """

    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.movies  = movies.set_index("title")
        self.ratings = ratings

        # Build feature string per movie
        feature_str = (
            movies["genres"] + " " +
            movies["director"] + " " +
            movies["year"].astype(str)
        )
        feature_str.index = movies["title"]

        tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z0-9]+")
        tfidf_matrix = tfidf.fit_transform(feature_str)
        self.sim = pd.DataFrame(
            cosine_similarity(tfidf_matrix),
            index=movies["title"],
            columns=movies["title"]
        )

    def recommend(self, user: str, top_n: int = 5) -> pd.Series:
        if user not in self.ratings.index:
            raise ValueError(f"User '{user}' not found.")

        user_ratings = self.ratings.loc[user]
        liked = user_ratings[user_ratings >= 4].index.tolist()
        unrated = user_ratings[user_ratings == 0].index.tolist()

        if not liked:
            raise ValueError(f"User '{user}' has no highly-rated movies to base recommendations on.")

        scores = {}
        for movie in unrated:
            if movie in self.sim.columns:
                scores[movie] = self.sim.loc[liked, movie].mean()

        return (pd.Series(scores)
                  .sort_values(ascending=False)
                  .head(top_n))


# ─── Hybrid Helper ────────────────────────────────────────────────────────────

def hybrid_recommend(user: str, cf: CollaborativeFilteringRecommender,
                     cb: ContentBasedRecommender,
                     cf_weight: float = 0.5, top_n: int = 5) -> pd.Series:
    """
    Combine CF and CB scores with configurable weights.
    """
    try:
        cf_scores = cf.recommend(user, top_n=20)
    except Exception:
        cf_scores = pd.Series(dtype=float)

    try:
        cb_scores = cb.recommend(user, top_n=20)
    except Exception:
        cb_scores = pd.Series(dtype=float)

    all_movies = set(cf_scores.index) | set(cb_scores.index)
    combined = {}
    for movie in all_movies:
        combined[movie] = (cf_weight * cf_scores.get(movie, 0) +
                           (1 - cf_weight) * cb_scores.get(movie, 0))

    return (pd.Series(combined)
              .sort_values(ascending=False)
              .head(top_n))


# ─── Demo ─────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")

def show_user_profile(user: str):
    rated = RATINGS.loc[user]
    rated = rated[rated > 0].sort_values(ascending=False)
    print(f"\n  Ratings by {user}:")
    for movie, rating in rated.items():
        stars = "⭐" * int(rating)
        print(f"    {movie:<35} {stars}")

def main():
    print("=" * 55)
    print("  🎬  Recommendation System  |  CodSoft AI Task 4")
    print("=" * 55)

    cf = CollaborativeFilteringRecommender(RATINGS)
    cb = ContentBasedRecommender(MOVIES, RATINGS)

    demo_users = ["Alice", "Charlie", "Eve", "Frank"]

    for user in demo_users:
        print_section(f"User: {user}")
        show_user_profile(user)

        # Collaborative
        print(f"\n  📊 Collaborative Filtering Recommendations:")
        try:
            cf_recs = cf.recommend(user, top_n=3)
            for i, (movie, score) in enumerate(cf_recs.items(), 1):
                print(f"    {i}. {movie:<35} (score: {score:.3f})")
        except Exception as e:
            print(f"    ⚠️  {e}")

        # Content-Based
        print(f"\n  🎯 Content-Based Recommendations:")
        try:
            cb_recs = cb.recommend(user, top_n=3)
            for i, (movie, score) in enumerate(cb_recs.items(), 1):
                print(f"    {i}. {movie:<35} (score: {score:.3f})")
        except Exception as e:
            print(f"    ⚠️  {e}")

        # Hybrid
        print(f"\n  🔀 Hybrid Recommendations (50/50):")
        try:
            h_recs = hybrid_recommend(user, cf, cb, cf_weight=0.5, top_n=3)
            for i, (movie, score) in enumerate(h_recs.items(), 1):
                print(f"    {i}. {movie:<35} (score: {score:.3f})")
        except Exception as e:
            print(f"    ⚠️  {e}")

    print("\n" + "=" * 55)
    print("  ✅  Done! Recommendation System Demo Complete.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()

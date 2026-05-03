"""
explainer.py
------------
Generates human-readable explanations for recommendations.  [NOVELTY COMPONENT]

Explanation types:
  - Content-based: "Because you liked X which shares genres/tags with Y"
  - Collaborative: "Users similar to you also enjoyed Y"
  - Hybrid: "Based on your taste profile (CF weight X%, CB weight Y%)"
  - Cold-start: "Since you're new, we matched your preferences to..."
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class Explainer:
    """
    Generates natural-language explanations for recommendations.

    Parameters
    ----------
    movies_df      : DataFrame with columns movieId, title_clean, genres_list
    content_filter : fitted ContentBasedFilter instance
    train_df       : training ratings DataFrame
    """

    def __init__(self, movies_df: pd.DataFrame, content_filter, train_df: pd.DataFrame):
        self._movies  = movies_df.set_index("movieId") if "movieId" in movies_df.columns else movies_df
        self._cb      = content_filter
        self._train   = train_df
        self._pop     = train_df.groupby("movieId")["userId"].count()  # popularity

    # ------------------------------------------------------------------
    def _title(self, movie_id: int) -> str:
        try:
            return self._movies.loc[movie_id, "title_clean"]
        except (KeyError, Exception):
            return f"Movie #{movie_id}"

    def _genres(self, movie_id: int) -> List[str]:
        try:
            g = self._movies.loc[movie_id, "genres_list"]
            return list(g) if g else []
        except (KeyError, Exception):
            return []

    # ------------------------------------------------------------------
    def explain_content(self, recommended_id: int,
                        seed_movie_id: int) -> str:
        """
        Explain a CB recommendation in terms of a seed movie.
        """
        rec_title  = self._title(recommended_id)
        seed_title = self._title(seed_movie_id)
        rec_genres  = self._genres(recommended_id)
        seed_genres = self._genres(seed_movie_id)
        shared = list(set(rec_genres) & set(seed_genres))

        rec_tags  = self._cb.get_top_tags(recommended_id,  top_n=3)
        seed_tags = self._cb.get_top_tags(seed_movie_id, top_n=3)
        shared_tags = list(set(rec_tags) & set(seed_tags))

        parts = [f'**{rec_title}** is recommended because you liked **{seed_title}**.']

        if shared:
            parts.append(f"They share genre(s): *{', '.join(shared)}*.")
        if shared_tags:
            parts.append(f"Common themes: *{', '.join(shared_tags)}*.")

        sim = self._cb.similarity_between(recommended_id, seed_movie_id)
        parts.append(f"Content similarity score: **{sim:.2f}**.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    def explain_cf(self, recommended_id: int, user_id: int) -> str:
        """
        Explain a CF recommendation.
        """
        rec_title   = self._title(recommended_id)
        rating_count = len(
            self._train[self._train["userId"] == user_id]
        )
        pop = int(self._pop.get(recommended_id, 0))
        return (
            f'**{rec_title}** is recommended because users with similar taste '
            f'to yours (based on your {rating_count} ratings) enjoyed it. '
            f'It has been rated by **{pop}** users in total.'
        )

    # ------------------------------------------------------------------
    def explain_hybrid(
        self,
        recommended_id: int,
        user_id: int,
        w_cf: float,
        w_cb: float,
        top_rated_ids: Optional[List[int]] = None,
    ) -> str:
        """
        Full hybrid explanation combining CF and CB signals.
        """
        rec_title = self._title(recommended_id)
        rec_genres = self._genres(recommended_id)

        lines = [f"### Why *{rec_title}*?"]

        # Weight explanation
        pct_cf = round(w_cf * 100)
        pct_cb = round(w_cb * 100)
        lines.append(
            f"Your recommendation mix: **{pct_cf}% collaborative** / **{pct_cb}% content-based**."
        )

        # CF signal
        if w_cf >= 0.3:
            pop = int(self._pop.get(recommended_id, 0))
            lines.append(
                f"🤝 *Collaborative signal*: Users similar to you rated this movie highly "
                f"({pop} total ratings in dataset)."
            )

        # CB signal — find most similar rated movie
        if w_cb >= 0.3 and top_rated_ids:
            best_seed = None
            best_sim  = -1
            for seed in top_rated_ids[:10]:   # check top 10 rated
                s = self._cb.similarity_between(recommended_id, seed)
                if s > best_sim:
                    best_sim  = s
                    best_seed = seed
            if best_seed and best_sim > 0.05:
                seed_title = self._title(best_seed)
                shared = list(set(self._genres(recommended_id)) & set(self._genres(best_seed)))
                lines.append(
                    f"🎬 *Content signal*: Similar to **{seed_title}** you enjoyed "
                    + (f"(shared genres: *{', '.join(shared[:3])}*)." if shared else "(shared themes).")
                )

        # Genre tags of recommended
        if rec_genres:
            lines.append(f"🏷️ *Genres*: {', '.join(rec_genres[:5])}.")

        # Popularity signal
        pop = int(self._pop.get(recommended_id, 0))
        if pop > 100:
            lines.append(f"📈 *Popularity*: Widely watched ({pop} ratings).")
        elif pop < 20:
            lines.append(f"💎 *Hidden gem*: Rarely seen but matches your taste ({pop} ratings).")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def explain_cold_start(self, recommended_id: int, preferred_genres: List[str]) -> str:
        rec_title  = self._title(recommended_id)
        rec_genres = self._genres(recommended_id)
        matched    = list(set(rec_genres) & set(preferred_genres))
        tags       = self._cb.get_top_tags(recommended_id, top_n=3)

        parts = [f"**{rec_title}** matches your selected preferences."]
        if matched:
            parts.append(f"Genre match: *{', '.join(matched)}*.")
        if tags:
            parts.append(f"Themes: *{', '.join(tags)}*.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    def profile_summary(self, user_id: int, top_n: int = 5) -> dict:
        """
        Summarise a user's taste profile from their ratings.
        Returns a dict with favourite genres, top-rated movies, avg rating.
        """
        user_ratings = self._train[self._train["userId"] == user_id].copy()
        if user_ratings.empty:
            return {"user_id": user_id, "status": "no_history"}

        user_ratings = user_ratings.merge(
            self._movies[["title_clean", "genres_list"]].reset_index(),
            on="movieId", how="left"
        )

        avg_rating = round(user_ratings["rating"].mean(), 2)

        # Genre frequency weighted by rating
        genre_weights: dict = {}
        for _, row in user_ratings.iterrows():
            g_list = row.get("genres_list", [])
            for g in (g_list if isinstance(g_list, list) else []):
                genre_weights[g] = genre_weights.get(g, 0) + row["rating"]

        top_genres = sorted(genre_weights.items(), key=lambda x: x[1], reverse=True)[:5]

        top_movies = (
            user_ratings.nlargest(top_n, "rating")[["movieId", "title_clean", "rating"]]
            .rename(columns={"title_clean": "title"})
            .to_dict("records")
        )

        return {
            "user_id":    user_id,
            "n_ratings":  len(user_ratings),
            "avg_rating": avg_rating,
            "top_genres": [g for g, _ in top_genres],
            "top_movies": top_movies,
        }

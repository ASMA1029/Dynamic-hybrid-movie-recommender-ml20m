"""
evaluation.py
-------------
Offline evaluation of recommender systems.
CS-825 Information Retrieval  ·  Spring 2026
Authors: Asma Bibi | Nimra Hashmi | Samia Jamil

Metrics implemented
───────────────────
  Rating Prediction Accuracy:
    RMSE    — Root Mean Squared Error
    MAE     — Mean Absolute Error

  Ranking Quality (computed at cutoff K):
    Precision@K   — fraction of top-K that are relevant
    Recall@K      — fraction of all relevant items retrieved
    F1@K          — harmonic mean of P and R
    NDCG@K        — Normalised Discounted Cumulative Gain

  Beyond-Accuracy (computed at cutoff K):
    Coverage      — % of catalogue the system recommends across all users
    Diversity@K   — intra-list diversity via avg pairwise dissimilarity
    Serendipity@K — relevant AND dissimilar to user history
    Novelty@K     — mean self-information (inverse popularity)

  Visualisation:
    plot_sigmoid_weights()  — CF/CB weight curve with user marker (FIXED)
    plot_metric_comparison() — bar chart comparing models across metrics
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional

# ── optional plotly import (only needed for chart functions) ──────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY = True
except ImportError:
    _PLOTLY = False


# ══════════════════════════════════════════════════════════════════════════════
# 1.  RATING PREDICTION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def rmse(test_df: pd.DataFrame, predict_fn: Callable) -> float:
    """
    Root Mean Squared Error.

    Iterates over (userId, movieId, rating) rows in test_df.
    Skips rows where the model returns NaN (can't predict).

    Formula:  RMSE = sqrt( mean( (r_hat - r)^2 ) )

    Parameters
    ----------
    test_df    : DataFrame with columns ['userId', 'movieId', 'rating']
    predict_fn : callable(user_id: int, movie_id: int) → float
    """
    squared_errors = []
    for _, row in test_df.iterrows():
        pred = predict_fn(int(row["userId"]), int(row["movieId"]))
        if pred is not None and not np.isnan(pred):
            squared_errors.append((pred - row["rating"]) ** 2)
    return float(np.sqrt(np.mean(squared_errors))) if squared_errors else np.nan


def mae(test_df: pd.DataFrame, predict_fn: Callable) -> float:
    """
    Mean Absolute Error.

    Formula:  MAE = mean( |r_hat - r| )
    """
    abs_errors = []
    for _, row in test_df.iterrows():
        pred = predict_fn(int(row["userId"]), int(row["movieId"]))
        if pred is not None and not np.isnan(pred):
            abs_errors.append(abs(pred - row["rating"]))
    return float(np.mean(abs_errors)) if abs_errors else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RANKING METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _relevant_items(test_df: pd.DataFrame,
                    user_id: int,
                    threshold: float = 3.5) -> set:
    """
    Return the set of movieIds that user_id rated >= threshold in the test set.
    These are treated as ground-truth 'relevant' items.

    threshold=3.5 means ratings of 4.0 or 5.0 count as relevant
    (MovieLens scale is 0.5 – 5.0 in half-star steps).
    """
    user_rows = test_df[test_df["userId"] == user_id]
    return set(user_rows[user_rows["rating"] >= threshold]["movieId"].tolist())


def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Precision@K = (# relevant items in top-K) / K

    Measures: of the K items we recommended, how many were actually good?
    """
    top_k = recommended[:k]
    hits  = sum(1 for m in top_k if m in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Recall@K = (# relevant items in top-K) / (total relevant items)

    Measures: of all good items the user liked, how many did we find?
    """
    top_k = recommended[:k]
    hits  = sum(1 for m in top_k if m in relevant)
    return hits / len(relevant) if relevant else 0.0


def f1_at_k(precision: float, recall: float) -> float:
    """
    F1@K = harmonic mean of Precision@K and Recall@K.
    Balances both P and R into a single score.
    """
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.

    Rewards finding relevant items AND placing them higher in the list.
    A hit at position 1 scores more than a hit at position 10.

    Formula:
        DCG  = Σ 1/log2(i+2)  for each relevant item at rank i (0-indexed)
        IDCG = DCG of perfect ranking (all relevant items first)
        NDCG = DCG / IDCG
    """
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, movie_id in enumerate(recommended[:k])
        if movie_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BEYOND-ACCURACY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def intra_list_diversity(recommended: List[int],
                         similarity_fn: Callable,
                         k: int) -> float:
    """
    Diversity@K = average pairwise dissimilarity within the top-K list.

    Higher score = more varied recommendations (genres, styles differ).
    Formula: mean( 1 - sim(i, j) ) for all pairs i≠j in top-K

    similarity_fn(movie_id_a, movie_id_b) → float in [0, 1]
    """
    top_k = recommended[:k]
    if len(top_k) < 2:
        return 0.0
    dissimilarities = [
        1.0 - similarity_fn(top_k[i], top_k[j])
        for i in range(len(top_k))
        for j in range(i + 1, len(top_k))
    ]
    return float(np.mean(dissimilarities))


def serendipity_at_k(
    recommended: List[int],
    relevant: set,
    user_rated_ids: List[int],
    similarity_fn: Callable,
    k: int,
    sim_threshold: float = 0.5,
) -> float:
    """
    Serendipity@K = proportion of top-K items that are BOTH:
        (a) relevant  — user actually rated it highly in the test set
        (b) unexpected — dissimilar to the user's past viewing history

    A serendipitous rec is one the user liked but wouldn't have found alone.

    Parameters
    ----------
    sim_threshold : avg similarity to history above this → NOT serendipitous
                    (default 0.5 means items must be less than 50% similar)
    """
    top_k = recommended[:k]
    serendipitous = 0
    for movie_id in top_k:
        if movie_id not in relevant:
            continue  # not relevant → can't be serendipitous
        if not user_rated_ids:
            serendipitous += 1  # no history → everything is unexpected
            continue
        avg_sim = float(np.mean([
            similarity_fn(movie_id, hist_movie)
            for hist_movie in user_rated_ids
        ]))
        if avg_sim <= sim_threshold:
            serendipitous += 1
    return serendipitous / k if k > 0 else 0.0


def novelty_at_k(recommended: List[int],
                 item_popularity: pd.Series,
                 k: int) -> float:
    """
    Novelty@K = mean self-information of recommended items.

    Measures how 'niche' the recommendations are.
    Higher score = less popular, more novel items recommended.

    Formula: mean( -log2( pop(i) / total_ratings ) ) for i in top-K

    item_popularity : pd.Series  indexed by movieId
                                  values   = number of ratings received
    """
    top_k      = recommended[:k]
    total      = item_popularity.sum()
    if total == 0:
        return np.nan
    scores = [
        -np.log2(item_popularity.get(m, 1) / total)
        for m in top_k
    ]
    return float(np.mean(scores)) if scores else np.nan


def coverage(all_recommendations: List[List[int]],
             catalogue_size: int) -> float:
    """
    Coverage = fraction of the full item catalogue that appears
               in at least one recommendation list across all users.

    Low coverage → the system keeps recommending the same popular items.
    High coverage → the system explores the full catalogue.
    """
    if catalogue_size == 0:
        return 0.0
    recommended_set = {m for rec_list in all_recommendations for m in rec_list}
    return len(recommended_set) / catalogue_size


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FULL EVALUATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model_name:          str,
    recommend_fn:        Callable,
    predict_fn:          Optional[Callable],
    train_df:            pd.DataFrame,
    test_df:             pd.DataFrame,
    movies_df:           pd.DataFrame,
    similarity_fn:       Optional[Callable] = None,
    k:                   int   = 14,
    relevance_threshold: float = 3.5,
    max_users:           int   = 85,
    rmse_sample_size:    int   = 1000,
    seed:                int   = 42,
) -> Dict:
    """
    Run comprehensive offline evaluation for one recommender model.

    How it works
    ────────────
    RMSE / MAE:
        Sampled from up to `rmse_sample_size` random rows in test_df.
        Skipped entirely if predict_fn is None (e.g. CBF, Hybrid).

    Ranking metrics (P@K, R@K, F1@K, NDCG@K, Diversity@K, Novelty@K):
        Computed across `max_users` randomly selected test users.
        Only users who have at least one relevant item in the test set
        are included in the averages.

    Coverage:
        Computed across ALL recommendation lists from the evaluated users.

    Parameters
    ──────────
    model_name          : label shown in results table
    recommend_fn        : callable(user_id, n) → list of (movie_id, score)
                          tuples OR plain list of movie_ids
    predict_fn          : callable(user_id, movie_id) → float rating prediction
                          pass None to skip RMSE/MAE (CBF, Hybrid)
    train_df            : training ratings DataFrame
    test_df             : test ratings DataFrame
    movies_df           : movies metadata DataFrame
    similarity_fn       : callable(movie_id_a, movie_id_b) → float [0,1]
                          pass None to skip Diversity and Serendipity
    k                   : recommendation list cutoff (default 14)
    relevance_threshold : min rating to count as relevant (default 3.5)
    max_users           : number of users to evaluate for ranking metrics
    rmse_sample_size    : max test rows for RMSE/MAE computation
    seed                : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # ── RMSE / MAE ────────────────────────────────────────────────────────────
    rmse_score = mae_score = np.nan
    if predict_fn is not None:
        n_sample    = min(rmse_sample_size, len(test_df))
        sample_test = test_df.sample(n=n_sample, random_state=seed)
        print(f"  [{model_name}] Computing RMSE/MAE on {n_sample} test rows ...")
        rmse_score  = rmse(sample_test, predict_fn)
        mae_score   = mae(sample_test,  predict_fn)
        print(f"  [{model_name}] RMSE={rmse_score:.4f}  MAE={mae_score:.4f}")

    # ── Select evaluation users ───────────────────────────────────────────────
    test_users = test_df["userId"].unique().copy()
    rng.shuffle(test_users)
    eval_users = test_users[:max_users]

    # Pre-compute item popularity from training set
    item_pop = train_df.groupby("movieId")["userId"].count()

    # Accumulators
    p_scores, r_scores, f1_scores, ndcg_scores = [], [], [], []
    div_scores, ser_scores, nov_scores = [], [], []
    all_rec_lists: List[List[int]] = []
    skipped = 0

    print(f"  [{model_name}] Evaluating {len(eval_users)} users at K={k} ...")

    for uid in eval_users:
        # Ground truth relevant items for this user
        relevant = _relevant_items(test_df, uid, relevance_threshold)
        if not relevant:
            skipped += 1
            continue  # user has no relevant items in test set — skip

        # Get recommendations
        try:
            rec_list = recommend_fn(uid, k)
        except Exception as e:
            print(f"  [{model_name}] recommend_fn failed for user {uid}: {e}")
            skipped += 1
            continue

        if not rec_list:
            skipped += 1
            continue

        # Normalise: accept (movie_id, score) tuples OR plain movie_id list
        if isinstance(rec_list[0], (tuple, list)):
            rec_ids = [int(item[0]) for item in rec_list]
        else:
            rec_ids = [int(m) for m in rec_list]

        all_rec_lists.append(rec_ids)

        # Ranking metrics
        p = precision_at_k(rec_ids, relevant, k)
        r = recall_at_k(rec_ids, relevant, k)
        p_scores.append(p)
        r_scores.append(r)
        f1_scores.append(f1_at_k(p, r))
        ndcg_scores.append(ndcg_at_k(rec_ids, relevant, k))

        # Diversity & Serendipity (require similarity function)
        if similarity_fn is not None:
            div_scores.append(intra_list_diversity(rec_ids, similarity_fn, k))
            user_history = train_df[train_df["userId"] == uid]["movieId"].tolist()
            ser_scores.append(
                serendipity_at_k(rec_ids, relevant, user_history, similarity_fn, k)
            )

        # Novelty
        nov_scores.append(novelty_at_k(rec_ids, item_pop, k))

    n_evaluated = len(p_scores)
    print(f"  [{model_name}] Evaluated: {n_evaluated} users  |  Skipped: {skipped}")

    # ── Coverage ──────────────────────────────────────────────────────────────
    catalogue_size = movies_df["movieId"].nunique()
    cov = coverage(all_rec_lists, catalogue_size)

    # ── Helper: safe mean ──────────────────────────────────────────────────────
    def _avg(lst):
        return round(float(np.mean(lst)), 4) if lst else np.nan

    def _fmt(val):
        """Format float or return 'N/A' string for NaN."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return round(float(val), 4)

    return {
        "Model":              model_name,
        "RMSE":               _fmt(rmse_score),
        "MAE":                _fmt(mae_score),
        f"P@{k}":             _avg(p_scores),
        f"R@{k}":             _avg(r_scores),
        f"F1@{k}":            _avg(f1_scores),
        f"NDCG@{k}":          _avg(ndcg_scores),
        "Coverage":           round(cov, 4),
        f"Diversity@{k}":     _avg(div_scores) if div_scores else "N/A",
        f"Serendipity@{k}":   _avg(ser_scores) if ser_scores else "N/A",
        f"Novelty@{k}":       _avg(nov_scores),
        "n_users_eval":       n_evaluated,
    }


def compare_models(results: List[Dict]) -> pd.DataFrame:
    """
    Convert a list of evaluate_model() result dicts into a
    formatted comparison DataFrame indexed by model name.

    Usage
    -----
    results = [
        evaluate_model("UserCF",  ...),
        evaluate_model("ItemCF",  ...),
        evaluate_model("SVD",     ...),
        evaluate_model("CBF",     ...),
        evaluate_model("Hybrid",  ...),
    ]
    df = compare_models(results)
    print(df.to_string())
    """
    return pd.DataFrame(results).set_index("Model")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION  (requires plotly)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sigmoid_weights(user_rating_count: int,
                         alpha: float = 0.15,
                         theta: int   = 20,
                         max_ratings: int = 150):
    """
    Plot the Dynamic Cold-Start Sigmoid weight curve.

    Shows the full CF weight curve from 0 → max_ratings,
    plus a red marker at the current user's position.

    FIX: Previously this was called with a scalar (e.g. rating_count=1)
    which only plotted a single dot.  This function always generates the
    full x-range array internally — you just pass the user's count.

    Parameters
    ----------
    user_rating_count : int   — how many ratings this user has
    alpha             : float — sigmoid steepness  (default 0.15)
    theta             : int   — equal-weight threshold (default 20)
    max_ratings       : int   — x-axis upper bound (default 150)

    Returns
    -------
    plotly Figure object (call fig.show() or pass to st.plotly_chart())

    Formula
    -------
    w_CF(u) = 1 / (1 + e^( -alpha * (|R_u| - theta) ))
    w_CB(u) = 1 - w_CF(u)
    """
    if not _PLOTLY:
        raise ImportError("plotly is required: pip install plotly")

    # ── Full curve array (0 to max_ratings) ───────────────────────────────────
    x    = np.arange(0, max_ratings + 1)
    w_cf = 1.0 / (1.0 + np.exp(-alpha * (x - theta)))
    w_cb = 1.0 - w_cf

    # ── Current user's position ────────────────────────────────────────────────
    user_x    = max(0, min(user_rating_count, max_ratings))
    user_wcf  = 1.0 / (1.0 + np.exp(-alpha * (user_x - theta)))
    user_wcb  = 1.0 - user_wcf

    fig = go.Figure()

    # CF weight curve (blue)
    fig.add_trace(go.Scatter(
        x=x, y=w_cf,
        mode="lines",
        name="CF Weight",
        line=dict(color="#2E86DE", width=3),
        hovertemplate="Ratings: %{x}<br>CF Weight: %{y:.3f}<extra></extra>"
    ))

    # CB weight curve (teal)
    fig.add_trace(go.Scatter(
        x=x, y=w_cb,
        mode="lines",
        name="CB Weight",
        line=dict(color="#17A589", width=3, dash="dash"),
        hovertemplate="Ratings: %{x}<br>CB Weight: %{y:.3f}<extra></extra>"
    ))

    # θ = 20 equal-weight dashed line
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#F39C12",
        line_width=1.5,
        annotation_text=f"θ={theta} Equal Split",
        annotation_position="top right",
        annotation_font_color="#F39C12"
    )

    # Current user marker (red dot)
    label = "Cold-Start" if user_x < theta else "Active User"
    fig.add_trace(go.Scatter(
        x=[user_x],
        y=[user_wcf],
        mode="markers+text",
        name=label,
        marker=dict(color="#E74C3C", size=12, symbol="circle",
                    line=dict(color="white", width=2)),
        text=[f"  {label}<br>  CF={user_wcf:.0%} CB={user_wcb:.0%}"],
        textposition="middle right",
        textfont=dict(color="#E74C3C", size=11),
        hovertemplate=(
            f"User has {user_x} ratings<br>"
            f"CF Weight: {user_wcf:.3f}<br>"
            f"CB Weight: {user_wcb:.3f}<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(
            text="CF Weight vs Rating Count — Dynamic Cold-Start Sigmoid",
            font=dict(size=15)
        ),
        xaxis=dict(
            title="Number of Ratings",
            range=[0, max_ratings],
            gridcolor="#333",
        ),
        yaxis=dict(
            title="Weight",
            range=[0, 1],
            tickformat=".0%",
            gridcolor="#333",
        ),
        legend=dict(x=0.75, y=0.5),
        template="plotly_dark",
        margin=dict(l=60, r=40, t=60, b=50),
        hovermode="x unified",
    )
    return fig


def plot_metric_comparison(results: List[Dict],
                           metrics: Optional[List[str]] = None,
                           k: int = 14):
    """
    Bar chart comparing all models across selected metrics.

    Parameters
    ----------
    results : list of evaluate_model() dicts
    metrics : list of metric names to include
              (default: RMSE, P@K, NDCG@K, Coverage, Novelty@K)
    k       : cutoff value used (for column name lookup)

    Returns
    -------
    plotly Figure
    """
    if not _PLOTLY:
        raise ImportError("plotly is required: pip install plotly")

    if metrics is None:
        metrics = ["RMSE", "MAE", f"P@{k}", f"NDCG@{k}",
                   "Coverage", f"Diversity@{k}", f"Novelty@{k}"]

    df = compare_models(results).reset_index()

    # Replace "N/A" strings with NaN for plotting
    for col in metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    COLORS = ["#2E86DE", "#17A589", "#E07B39", "#8E44AD", "#F39C12"]

    fig = go.Figure()
    for i, row in df.iterrows():
        vals  = [row.get(m, np.nan) for m in metrics]
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=metrics,
            y=vals,
            marker_color=color,
            text=[f"{v:.4f}" if not np.isnan(v) else "N/A" for v in vals],
            textposition="outside",
        ))

    fig.update_layout(
        title="Model Comparison — All Metrics",
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Score",
        template="plotly_dark",
        legend=dict(x=1.01, y=1),
        margin=dict(l=60, r=160, t=60, b=80),
    )
    return fig

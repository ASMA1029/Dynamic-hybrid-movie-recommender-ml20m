"""
collaborative.py — Collaborative Filtering (no scikit-surprise)
Memory-safe: trims pivot table to top users/movies before building matrix

Authors: Asma Bibi | Nimra Hashmi | Samia Jamil
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

K_NEIGHBOURS    = 20
MIN_CO_RATINGS  = 3
K_FACTORS       = 50
N_EPOCHS        = 20
LR              = 0.005
REG             = 0.02
MAX_MATRIX_CELLS = 4_000_000   # hard cap — 4M cells max


def _safe_pivot(train_df, max_cells=MAX_MATRIX_CELLS):
    """Build pivot table — auto-trim if too large for RAM."""
    n_users  = train_df['userId'].nunique()
    n_movies = train_df['movieId'].nunique()

    if n_users * n_movies > max_cells:
        print(f"[CF] Matrix {n_users}x{n_movies} too large — trimming ...")
        top_users  = train_df.groupby('userId').size().nlargest(1500).index
        top_movies = train_df.groupby('movieId').size().nlargest(2000).index
        train_df   = train_df[
            train_df['userId'].isin(top_users) &
            train_df['movieId'].isin(top_movies)
        ]
        print(f"[CF] Trimmed to {train_df['userId'].nunique()}x{train_df['movieId'].nunique()}")

    return train_df.pivot_table(index='userId', columns='movieId', values='rating')


# ═══════════════════════════════════════════════════════
# 1. USER-BASED CF
# ═══════════════════════════════════════════════════════
class UserCF:
    def __init__(self, k=K_NEIGHBOURS, min_co=MIN_CO_RATINGS, min_common=None):
        self.k      = k
        self.min_co = min_common if min_common is not None else min_co
        self.matrix = self.sim = self.means = None
        self.users  = []; self.movies = []; self.u_idx = {}

    def fit(self, train_df):
        print("[UserCF] Fitting ...")
        self.matrix = _safe_pivot(train_df)
        self.users  = self.matrix.index.tolist()
        self.movies = self.matrix.columns.tolist()
        self.u_idx  = {u: i for i, u in enumerate(self.users)}
        self.means  = self.matrix.mean(axis=1)
        centred     = self.matrix.sub(self.means, axis=0).fillna(0).values
        self.sim    = cosine_similarity(centred)
        print(f"[UserCF] Done. {self.sim.shape}")

    def predict(self, user_id, movie_id):
        if user_id not in self.u_idx or movie_id not in self.matrix.columns:
            return float(self.means.mean()) if self.means is not None else 3.5
        u_i = self.u_idx[user_id]; mu_u = self.means[user_id]
        sims = self.sim[u_i].copy(); sims[u_i] = 0
        rated_mask  = self.matrix[movie_id].notna().values
        sims_masked = sims * rated_mask
        top_k_idx   = np.argsort(sims_masked)[::-1][:self.k]
        top_k_sim   = sims_masked[top_k_idx]
        valid = top_k_sim > 0
        if not valid.any(): return float(mu_u)
        top_k_idx = top_k_idx[valid]; top_k_sim = top_k_sim[valid]
        top_users = [self.users[i] for i in top_k_idx]
        num = sum(top_k_sim[j] * (self.matrix.loc[v, movie_id] - self.means[v])
                  for j, v in enumerate(top_users) if pd.notna(self.matrix.loc[v, movie_id]))
        den = np.sum(np.abs(top_k_sim))
        return float(np.clip(mu_u + (num/den if den else 0), 0.5, 5.0))

    def recommend(self, user_id, n=10, candidate_ids=None):
        if user_id not in self.u_idx:
            return []
        rated = set(self.matrix.columns[self.matrix.loc[user_id].notna()]) if user_id in self.matrix.index else set()
        if candidate_ids is None:
            candidate_ids = [m for m in self.movies if m not in rated]
        scores = [(m, self.predict(user_id, m)) for m in candidate_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


# ═══════════════════════════════════════════════════════
# 2. ITEM-BASED CF
# ═══════════════════════════════════════════════════════
class ItemCF:
    def __init__(self, k=K_NEIGHBOURS, min_common=None):
        self.k      = k
        self.matrix = self.sim = self.means = None
        self.items  = []; self.i_idx = {}

    def fit(self, train_df):
        print("[ItemCF] Fitting ...")
        self.matrix = _safe_pivot(train_df)
        self.items  = self.matrix.columns.tolist()
        self.i_idx  = {m: i for i, m in enumerate(self.items)}
        self.means  = self.matrix.mean(axis=1)
        centred     = self.matrix.sub(self.means, axis=0).fillna(0)
        self.sim    = cosine_similarity(centred.T.values)
        print(f"[ItemCF] Done. {self.sim.shape}")

    def predict(self, user_id, movie_id):
        if movie_id not in self.i_idx or user_id not in self.matrix.index:
            return 3.5
        m_i = self.i_idx[movie_id]
        sims = self.sim[m_i].copy(); sims[m_i] = 0
        user_ratings = self.matrix.loc[user_id]
        rated_mask   = user_ratings.notna().values
        sims_masked  = sims * rated_mask
        top_k_idx    = np.argsort(sims_masked)[::-1][:self.k]
        top_k_sim    = sims_masked[top_k_idx]
        valid = top_k_sim > 0
        if not valid.any(): return float(user_ratings.mean())
        top_k_idx  = top_k_idx[valid]; top_k_sim = top_k_sim[valid]
        top_movies = [self.items[i] for i in top_k_idx]
        num = sum(top_k_sim[j] * user_ratings[m]
                  for j, m in enumerate(top_movies) if pd.notna(user_ratings.get(m)))
        den = np.sum(np.abs(top_k_sim))
        return float(np.clip((num/den if den else user_ratings.mean()), 0.5, 5.0))

    def recommend(self, user_id, n=10, candidate_ids=None):
        if user_id not in self.matrix.index:
            return []
        rated = set(self.matrix.columns[self.matrix.loc[user_id].notna()])
        if candidate_ids is None:
            candidate_ids = [m for m in self.items if m not in rated]
        scores = [(m, self.predict(user_id, m)) for m in candidate_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


# ═══════════════════════════════════════════════════════
# 3. SVD (scipy — no scikit-surprise)
# ═══════════════════════════════════════════════════════
class SVDModel:
    def __init__(self, n_factors=K_FACTORS, n_epochs=N_EPOCHS, lr=LR, reg=REG, k=None):
        self.k        = n_factors if k is None else k
        self.n_epochs = n_epochs
        self.lr       = lr
        self.reg      = reg
        self.global_mean = 3.5
        self.user_bias = {}; self.item_bias = {}
        self.P = self.Q = None
        self.u_idx = {}; self.i_idx = {}
        self.users = []; self.items = []
        self._train_df = None

    def fit(self, train_df):
        print(f"[SVD] Fitting k={self.k} ...")
        self._train_df   = train_df
        self.users = sorted(train_df['userId'].unique())
        self.items = sorted(train_df['movieId'].unique())
        self.u_idx = {u: i for i, u in enumerate(self.users)}
        self.i_idx = {m: i for i, m in enumerate(self.items)}
        self.global_mean = train_df['rating'].mean()

        rows = train_df['userId'].map(self.u_idx).values
        cols = train_df['movieId'].map(self.i_idx).values
        vals = (train_df['rating'] - self.global_mean).values.astype(np.float32)
        n_u, n_i = len(self.users), len(self.items)
        sparse_mat = csr_matrix((vals, (rows, cols)), shape=(n_u, n_i))

        k_actual = min(self.k, min(n_u, n_i) - 1)
        print(f"[SVD] svds {n_u}x{n_i} k={k_actual} ...")
        U, sigma, Vt = svds(sparse_mat, k=k_actual)
        self.P = U  * np.sqrt(sigma)
        self.Q = Vt.T * np.sqrt(sigma)

        self.user_bias = {u: 0.0 for u in self.users}
        self.item_bias = {m: 0.0 for m in self.items}

        print(f"[SVD] Bias SGD {self.n_epochs} epochs ...")
        records = list(zip(train_df['userId'].values,
                           train_df['movieId'].values,
                           train_df['rating'].values))
        for epoch in range(self.n_epochs):
            np.random.shuffle(records)
            loss = 0.0
            for uid, mid, r in records:
                if uid not in self.u_idx or mid not in self.i_idx: continue
                u_i = self.u_idx[uid]; i_i = self.i_idx[mid]
                pred = (self.global_mean + self.user_bias[uid]
                        + self.item_bias[mid] + np.dot(self.P[u_i], self.Q[i_i]))
                e = r - pred; loss += e*e
                self.user_bias[uid] += self.lr*(e - self.reg*self.user_bias[uid])
                self.item_bias[mid] += self.lr*(e - self.reg*self.item_bias[mid])
            if (epoch+1) % 5 == 0:
                print(f"[SVD] Epoch {epoch+1}/{self.n_epochs} RMSE={np.sqrt(loss/len(records)):.4f}")
        print("[SVD] Done.")

    def predict(self, user_id, movie_id):
        u_b = self.user_bias.get(user_id, 0.0)
        i_b = self.item_bias.get(movie_id, 0.0)
        dot = 0.0
        if user_id in self.u_idx and movie_id in self.i_idx:
            dot = np.dot(self.P[self.u_idx[user_id]], self.Q[self.i_idx[movie_id]])
        return float(np.clip(self.global_mean + u_b + i_b + dot, 0.5, 5.0))

    def recommend(self, user_id, n=10, candidate_ids=None):
        if candidate_ids is None:
            candidate_ids = self.items
        if user_id not in self.u_idx:
            scores = [(m, self.global_mean + self.item_bias.get(m,0.0)) for m in candidate_ids]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:n]
        u_i = self.u_idx[user_id]; p_u = self.P[u_i]
        valid   = [(m, self.i_idx[m]) for m in candidate_ids if m in self.i_idx]
        unknown = [m for m in candidate_ids if m not in self.i_idx]
        scores  = []
        if valid:
            mids, i_indices = zip(*valid)
            Q_sub  = self.Q[list(i_indices)]
            dots   = Q_sub @ p_u
            i_bias = np.array([self.item_bias.get(m,0.0) for m in mids])
            preds  = np.clip(self.global_mean + self.user_bias.get(user_id,0.0) + i_bias + dots, 0.5, 5.0)
            scores.extend(zip(mids, preds.tolist()))
        for m in unknown:
            scores.append((m, self.global_mean + self.item_bias.get(m,0.0)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def rating_count(self, user_id):
        if self._train_df is None: return 0
        return int((self._train_df['userId'] == user_id).sum())

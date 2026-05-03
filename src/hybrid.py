"""
hybrid.py — Dynamic Hybrid Recommender
Matches app.py usage:
  hybrid = HybridRecommender(alpha=0.15, threshold=20, svd_weight=0.6)
  hybrid.attach(user_cf, item_cf, svd, cb, all_movie_ids)
  hybrid.recommend(user_id, n, rated_ids, ratings_series) -> list of (mid, score, w_cf, w_cb)
  hybrid.explain_weights(user_id) -> dict with w_cf, w_cb, reason, profile

Authors: Asma Bibi | Nimra Hashmi | Samia Jamil
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _normalize(scores):
    """Min-max normalize list of (id, score) to [0,1]."""
    if not scores:
        return {}
    vals = [s for _, s in scores]
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {mid: 0.5 for mid, _ in scores}
    return {mid: (s - mn) / (mx - mn) for mid, s in scores}


class HybridRecommender:
    def __init__(self, alpha=0.15, threshold=20, svd_weight=0.6):
        self.alpha      = alpha
        self.threshold  = threshold
        self.svd_weight = svd_weight
        self.ucf_weight = (1 - svd_weight) / 2
        self.icf_weight = (1 - svd_weight) / 2
        self.user_cf    = None
        self.item_cf    = None
        self.svd        = None
        self.cb         = None
        self.all_movie_ids = []

    def attach(self, user_cf, item_cf, svd_model, content_filter, all_movie_ids):
        self.user_cf       = user_cf
        self.item_cf       = item_cf
        self.svd           = svd_model
        self.cb            = content_filter
        self.all_movie_ids = all_movie_ids

    def _cf_weight(self, user_id):
        if self.svd is not None:
            n = self.svd.rating_count(user_id)
        else:
            n = 0
        return _sigmoid(self.alpha * (n - self.threshold))

    def recommend(self, user_id, n=10, rated_ids=None, ratings_series=None):
        """
        Returns list of (movieId, score, w_cf, w_cb)
        """
        if rated_ids is None:
            rated_ids = []
        rated_set  = set(rated_ids)
        candidates = [m for m in self.all_movie_ids if m not in rated_set]

        w_cf = self._cf_weight(user_id)
        w_cb = 1.0 - w_cf

        # ── CF scores ──────────────────────────────────────
        svd_scores  = dict(self.svd.recommend(user_id, n=len(candidates), candidate_ids=candidates))
        ucf_scores  = dict(self.user_cf.recommend(user_id, n=len(candidates), candidate_ids=candidates))
        icf_scores  = dict(self.item_cf.recommend(user_id, n=len(candidates), candidate_ids=candidates))

        # Blend CF
        cf_raw = {}
        for m in candidates:
            s = (self.svd_weight  * svd_scores.get(m, 0.0) +
                 self.ucf_weight  * ucf_scores.get(m, 0.0) +
                 self.icf_weight  * icf_scores.get(m, 0.0))
            cf_raw[m] = s

        # ── CB scores ──────────────────────────────────────
        if ratings_series is not None and len(rated_ids) > 0:
            cb_list = self.cb.recommend_for_user(
                rated_ids, ratings_series, n=len(candidates), candidate_ids=candidates
            )
        else:
            cb_list = [(m, 0.0) for m in candidates]
        cb_raw = dict(cb_list)

        # ── Normalise ──────────────────────────────────────
        cf_norm = _normalize(list(cf_raw.items()))
        cb_norm = _normalize(list(cb_raw.items()))

        # ── Fuse ───────────────────────────────────────────
        fused = {}
        for m in candidates:
            fused[m] = w_cf * cf_norm.get(m, 0.0) + w_cb * cb_norm.get(m, 0.0)

        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:n]
        return [(mid, score, w_cf, w_cb) for mid, score in top]

    def explain_weights(self, user_id):
        w_cf = self._cf_weight(user_id)
        w_cb = 1.0 - w_cf
        if self.svd is not None:
            n = self.svd.rating_count(user_id)
        else:
            n = 0

        if n == 0:
            profile = "cold-start"
            reason  = f"New user (0 ratings) — content-based dominates ({w_cb*100:.0f}% CB)"
        elif n < 5:
            profile = "cold-start"
            reason  = f"Very few ratings ({n}) — mostly content-based ({w_cb*100:.0f}% CB)"
        elif n < 20:
            profile = "warming-up"
            reason  = f"Warming up ({n} ratings) — mixed strategy"
        elif n < 50:
            profile = "warm"
            reason  = f"Active user ({n} ratings) — collaborative filtering leads ({w_cf*100:.0f}% CF)"
        else:
            profile = "experienced"
            reason  = f"Experienced user ({n} ratings) — SVD-dominant ({w_cf*100:.0f}% CF)"

        return {
            "w_cf":    round(w_cf, 4),
            "w_cb":    round(w_cb, 4),
            "n":       n,
            "profile": profile,
            "reason":  reason,
        }

"""
content_based.py — Content-Based Filtering
Matches all attributes app.py accesses:
  cb = ContentBasedFilter(max_features=5000, ngram_range=(1,2))
  cb.fit(movies_df)
  cb.recommend(movie_id, n)              -> list of (movieId, score)
  cb.recommend_for_user(movie_ids, ratings_series, n) -> list of (movieId, score)
  cb.similarity_between(id_a, id_b)     -> float
  cb._tfidf                              (vectorizer)
  cb._tfidf_mat                          (matrix)
  cb._id2idx                             (dict)

Authors: Asma Bibi | Nimra Hashmi | Samia Jamil
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_GENOME_TAGS  = 20
GENOME_THRESHOLD = 0.5


class ContentBasedFilter:
    def __init__(self, max_features=5000, ngram_range=(1,2)):
        self.max_features  = max_features
        self.ngram_range   = ngram_range
        # Public attributes app.py accesses directly
        self._tfidf        = None   # fitted TfidfVectorizer
        self._tfidf_mat    = None   # sparse matrix (n_movies x n_features)
        self._id2idx       = {}     # movieId -> row index
        self._idx2id       = []     # row index -> movieId
        self._movies_df    = None
        self.using_genome  = False

    def fit(self, movies_df, ratings_df=None, tags_df=None,
            genome_scores_df=None, genome_tags_df=None):
        print("[CBF] Fitting ...")
        movies = movies_df.copy()

        # Ensure required columns exist
        if 'genres' not in movies.columns:
            movies['genres'] = ''
        if 'title_clean' not in movies.columns:
            movies['title_clean'] = movies['title'].str.replace(r'\s*\(\d{4}\)\s*$','',regex=True).str.strip()
        if 'genres_list' not in movies.columns:
            movies['genres_list'] = movies['genres'].apply(
                lambda g: [x.strip() for x in g.split('|') if x.strip() and x != '(no genres listed)']
                if isinstance(g, str) else []
            )

        # Genre string
        movies['genre_str'] = (movies['genres']
                               .str.replace('|',' ',regex=False)
                               .str.replace('(no genres listed)','',regex=False)
                               .str.lower().str.strip())

        # User tags
        user_tag_str = pd.Series('', index=movies['movieId'])
        if tags_df is not None and not tags_df.empty and 'tag' in tags_df.columns:
            agg = (tags_df.dropna(subset=['tag'])
                   .assign(tag=lambda df: df['tag'].str.lower().str.strip())
                   .drop_duplicates(subset=['movieId','tag'])
                   .groupby('movieId')['tag']
                   .apply(lambda t: ' '.join(t)))
            user_tag_str = user_tag_str.add(agg, fill_value='')

        # Genome tags
        genome_str = pd.Series('', index=movies['movieId'])
        if genome_scores_df is not None and genome_tags_df is not None and not genome_scores_df.empty:
            self.using_genome = True
            tag_lookup = genome_tags_df.set_index('tagId')['tag'].to_dict()
            filtered   = genome_scores_df[genome_scores_df['relevance'] >= GENOME_THRESHOLD].copy()
            top_tags   = (filtered.sort_values('relevance', ascending=False)
                          .groupby('movieId').head(TOP_GENOME_TAGS))
            top_tags   = top_tags.copy()
            top_tags['tag_name'] = (top_tags['tagId'].map(tag_lookup)
                                    .fillna('').str.lower()
                                    .str.replace(' ','_',regex=False))
            genome_agg = (top_tags.groupby('movieId')['tag_name']
                          .apply(lambda t: ' '.join(t[t != ''])))
            genome_str = genome_str.add(genome_agg, fill_value='')

        # Build content soup
        movies = movies.set_index('movieId')
        movies['user_tag_str'] = user_tag_str.reindex(movies.index, fill_value='')
        movies['genome_str']   = genome_str.reindex(movies.index, fill_value='')
        movies['content_soup'] = (movies['genre_str'] + ' ' +
                                  movies['user_tag_str'] + ' ' +
                                  movies['genome_str']).str.strip()

        # TF-IDF
        self._tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            sublinear_tf=True,
        )
        self._tfidf_mat = self._tfidf.fit_transform(movies['content_soup'].fillna(''))
        self._idx2id    = movies.index.tolist()
        self._id2idx    = {mid: i for i, mid in enumerate(self._idx2id)}
        self._movies_df = movies.reset_index()

        print(f"[CBF] TF-IDF: {self._tfidf_mat.shape} genome={self.using_genome}")

    def _build_profile(self, movie_ids, ratings_series=None):
        """Build a user profile vector from a list of rated movie ids."""
        valid = [m for m in movie_ids if m in self._id2idx]
        if not valid:
            return None
        indices = [self._id2idx[m] for m in valid]
        vecs    = self._tfidf_mat[indices]
        if ratings_series is not None:
            weights = np.array([float(ratings_series.get(m, 3.5)) for m in valid])
            r_min, r_max = weights.min(), weights.max()
            if r_max > r_min:
                weights = (weights - r_min) / (r_max - r_min)
            else:
                weights = np.ones(len(valid))
        else:
            weights = np.ones(len(valid))
        profile = np.asarray(vecs.multiply(weights[:,np.newaxis]).sum(axis=0))
        norm    = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def recommend(self, movie_id_or_user_id, n=10, candidate_ids=None):
        """
        Dual mode:
          - If argument is a movieId in our index → find similar movies
          - Otherwise treated as movie_id for similarity search
        """
        mid = movie_id_or_user_id
        if mid not in self._id2idx:
            return []
        idx    = self._id2idx[mid]
        vec    = self._tfidf_mat[idx]
        scores = cosine_similarity(vec, self._tfidf_mat).flatten()
        scores[idx] = -1
        if candidate_ids is not None:
            mask = np.zeros(len(self._idx2id))
            for m in candidate_ids:
                if m in self._id2idx:
                    mask[self._id2idx[m]] = 1
            scores = scores * mask
        top_idx = np.argsort(scores)[::-1][:n]
        return [(self._idx2id[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def recommend_for_user(self, movie_ids, ratings_series=None, n=10, candidate_ids=None):
        """Recommend based on a user's rated movie list."""
        profile = self._build_profile(movie_ids, ratings_series)
        if profile is None:
            return []
        rated_set = set(movie_ids)
        scores    = cosine_similarity(profile, self._tfidf_mat).flatten()
        if candidate_ids is not None:
            all_ids = candidate_ids
        else:
            all_ids = [m for m in self._idx2id if m not in rated_set]
        scored = [(m, float(scores[self._id2idx[m]])) for m in all_ids if m in self._id2idx]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def similarity_between(self, movie_id_a, movie_id_b):
        """Cosine similarity between two movies — used by evaluation."""
        if movie_id_a not in self._id2idx or movie_id_b not in self._id2idx:
            return 0.0
        a = self._tfidf_mat[self._id2idx[movie_id_a]]
        b = self._tfidf_mat[self._id2idx[movie_id_b]]
        return float(cosine_similarity(a, b)[0][0])

    def get_top_features(self, movie_id, n=5):
        if movie_id not in self._id2idx:
            return []
        idx        = self._id2idx[movie_id]
        vec        = np.asarray(self._tfidf_mat[idx].todense()).flatten()
        feat_names = self._tfidf.get_feature_names_out()
        top_idx    = np.argsort(vec)[::-1][:n]
        return [feat_names[i] for i in top_idx if vec[i] > 0]

    def get_top_tags(self, movie_id, top_n=5):
        """Alias for get_top_features — used by explainer.py"""
        return self.get_top_features(movie_id, n=top_n)

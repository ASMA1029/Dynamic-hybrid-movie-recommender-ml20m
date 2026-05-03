"""
data_loader.py — Data Loading & Preprocessing Module
Memory-Safe Version with all aliases app.py expects

Authors: Asma Bibi | Nimra Hashmi | Samia Jamil
"""

import os
import pandas as pd
import numpy as np

RATINGS_SAMPLE_SIZE = 50_000
RANDOM_SEED         = 42
TEST_SIZE           = 0.20
MAX_MATRIX_CELLS    = 5_000_000


def _find_file(data_dir, *names):
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return None


def _fix_cols(df, mapping):
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in mapping:
            col_map[c] = mapping[cl]
    return df.rename(columns=col_map)


def load_data(data_dir='data', sample_ratings=RATINGS_SAMPLE_SIZE):
    print(f"[Loader] Loading from: {data_dir}")

    # 1. Ratings
    path = _find_file(data_dir, 'ratings.csv', 'rating.csv')
    if not path:
        raise FileNotFoundError(f"ratings.csv not found in '{data_dir}'. Place your data files in the 'data/' folder.")

    print("[Loader] Reading ratings.csv ...")
    ratings = pd.read_csv(path)
    ratings = _fix_cols(ratings, {'userid':'userId','movieid':'movieId','rating':'rating','timestamp':'timestamp'})
    keep = [c for c in ['userId','movieId','rating','timestamp'] if c in ratings.columns]
    ratings = ratings[keep]

    if sample_ratings and len(ratings) > sample_ratings:
        print(f"[Loader] Sampling {sample_ratings:,} from {len(ratings):,} ...")
        ratings = ratings.sample(n=sample_ratings, random_state=RANDOM_SEED)

    if 'timestamp' in ratings.columns:
        ratings = (ratings.sort_values('timestamp', ascending=False)
                          .drop_duplicates(subset=['userId','movieId'])
                          .reset_index(drop=True))
    else:
        ratings = ratings.drop_duplicates(subset=['userId','movieId']).reset_index(drop=True)

    print(f"[Loader] Ratings: {len(ratings):,} | Users: {ratings['userId'].nunique():,} | Movies: {ratings['movieId'].nunique():,}")

    # 2. Movies
    path = _find_file(data_dir, 'movies.csv', 'movie.csv')
    if not path:
        raise FileNotFoundError(f"movies.csv not found in '{data_dir}'.")
    movies = pd.read_csv(path)
    movies = _fix_cols(movies, {'movieid':'movieId','title':'title','genres':'genres'})
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    # title_clean: strip year from title
    movies['title_clean'] = movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()
    # genres_list: split genres into list
    movies['genres_list'] = movies['genres'].apply(
        lambda g: [x.strip() for x in g.split('|') if x.strip() and x != '(no genres listed)']
        if isinstance(g, str) else []
    )
    print(f"[Loader] Movies: {len(movies):,}")

    # 3. Tags
    path = _find_file(data_dir, 'tags.csv', 'tag.csv')
    tags = pd.DataFrame(columns=['userId','movieId','tag','timestamp'])
    if path:
        tags = pd.read_csv(path)
        tags = _fix_cols(tags, {'userid':'userId','movieid':'movieId','tag':'tag','timestamp':'timestamp'})
        if 'tag' in tags.columns:
            tags = tags.dropna(subset=['tag'])
            tags['tag'] = tags['tag'].str.lower().str.strip()
            tags = tags.drop_duplicates(subset=['userId','movieId','tag'])
        print(f"[Loader] Tags: {len(tags):,}")
    else:
        print("[Loader] tags.csv not found")

    # 4. Genome Scores
    path = _find_file(data_dir, 'genome-scores.csv', 'genome_scores.csv')
    genome_scores = None
    if path:
        print("[Loader] Loading genome-scores.csv ...")
        genome_scores = pd.read_csv(path)
        genome_scores = _fix_cols(genome_scores, {'movieid':'movieId','tagid':'tagId','relevance':'relevance'})
        rated = set(ratings['movieId'].unique())
        genome_scores = genome_scores[genome_scores['movieId'].isin(rated)]
        print(f"[Loader] Genome scores: {len(genome_scores):,}")
    else:
        print("[Loader] genome-scores.csv not found")

    # 5. Genome Tags
    path = _find_file(data_dir, 'genome-tags.csv', 'genome_tags.csv')
    genome_tags = None
    if path:
        genome_tags = pd.read_csv(path)
        genome_tags = _fix_cols(genome_tags, {'tagid':'tagId','tag':'tag'})
        print(f"[Loader] Genome tags: {len(genome_tags):,}")

    # 6. Train/Test Split
    print("[Loader] Splitting 80/20 ...")
    user_counts   = ratings.groupby('userId').size()
    solo_users    = user_counts[user_counts == 1].index
    multi_users   = user_counts[user_counts  > 1].index
    solo_ratings  = ratings[ratings['userId'].isin(solo_users)]
    multi_ratings = ratings[ratings['userId'].isin(multi_users)]

    train_parts, test_parts = [solo_ratings], []
    for uid, grp in multi_ratings.groupby('userId'):
        n_test    = max(1, int(len(grp) * TEST_SIZE))
        test_idx  = grp.sample(n=n_test, random_state=RANDOM_SEED).index
        train_idx = grp.index.difference(test_idx)
        train_parts.append(grp.loc[train_idx])
        test_parts.append(grp.loc[test_idx])

    ratings_train = pd.concat(train_parts).reset_index(drop=True)
    ratings_test  = (pd.concat(test_parts).reset_index(drop=True)
                     if test_parts else pd.DataFrame(columns=ratings.columns))
    print(f"[Loader] Train: {len(ratings_train):,} | Test: {len(ratings_test):,}")

    # 7. User-Item Matrix — memory safe
    print("[Loader] Building user-item matrix ...")
    n_users  = ratings_train['userId'].nunique()
    n_movies = ratings_train['movieId'].nunique()

    if n_users * n_movies > MAX_MATRIX_CELLS:
        print(f"[Loader] Matrix too large — trimming ...")
        top_users  = ratings_train.groupby('userId').size().nlargest(2000).index
        top_movies = ratings_train.groupby('movieId').size().nlargest(2500).index
        trimmed    = ratings_train[
            ratings_train['userId'].isin(top_users) &
            ratings_train['movieId'].isin(top_movies)
        ]
        print(f"[Loader] Trimmed: {trimmed['userId'].nunique():,} users x {trimmed['movieId'].nunique():,} movies")
        user_item_matrix = trimmed.pivot_table(index='userId', columns='movieId', values='rating')
    else:
        user_item_matrix = ratings_train.pivot_table(index='userId', columns='movieId', values='rating')

    sparsity = 1 - ratings_train.shape[0] / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
    print(f"[Loader] Matrix: {user_item_matrix.shape} | Sparsity: {sparsity:.1%}")

    result = {
        'ratings_train':    ratings_train,
        'ratings_test':     ratings_test,
        'movies':           movies,
        'tags':             tags,
        'genome_scores':    genome_scores,
        'genome_tags':      genome_tags,
        'user_item_matrix': user_item_matrix,
        # ── Aliases that app.py uses ──
        'train_df':         ratings_train,
        'test_df':          ratings_test,
        'movies_df':        movies,
        'tags_df':          tags,
        'ratings':          ratings_train,
        'ratings_df':       ratings_train,
    }
    return result


def get_dataset_stats(data):
    rt = data['ratings_train']
    return {
        'total_ratings':  len(rt),
        'n_users':        rt['userId'].nunique(),
        'n_movies':       rt['movieId'].nunique(),
        'mean_rating':    round(rt['rating'].mean(), 2),
        'sparsity':       round(1 - len(rt) / (rt['userId'].nunique() * rt['movieId'].nunique()), 4),
        'genome_enabled': data['genome_scores'] is not None,
        'n_genome_tags':  len(data['genome_tags']) if data['genome_tags'] is not None else 0,
    }


load_and_prepare = load_data

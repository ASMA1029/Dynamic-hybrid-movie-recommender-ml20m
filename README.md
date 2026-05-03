# 🎬 Hybrid Movie Recommender System
**Information Retrieval Course Project**  
Asma · Nimra · Samia | MovieLens ml-latest-small

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
1. Go to https://grouplens.org/datasets/movielens/
2. Download **ml-latest-small.zip**
3. Extract and place these files in the `data/` folder:
   - `ratings.csv`
   - `movies.csv`
   - `tags.csv`
   - `links.csv`

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🏗️ Project Structure

```
movie_recommender/
├── data/                    # Place MovieLens CSVs here
├── src/
│   ├── data_loader.py       # Data loading, preprocessing, train/test split
│   ├── collaborative.py     # UserCF, ItemCF, SVD (matrix factorisation)
│   ├── content_based.py     # TF-IDF content-based filtering
│   ├── hybrid.py            # Dynamic cold-start-aware hybrid fusion
│   ├── evaluation.py        # RMSE, MAE, P@K, R@K, Diversity, Serendipity
│   └── explainer.py         # Natural language explanations
├── app.py                   # Streamlit UI
├── requirements.txt
└── README.md
```

---

## ✨ Novelty Contributions

| Feature | Description |
|---------|-------------|
| **SVD Matrix Factorisation** | Latent factor model via `scikit-surprise`, better than plain cosine similarity |
| **Dynamic Hybrid Weights** | CF/CB weight shifts with user activity via sigmoid: cold users → CB dominates, warm users → CF dominates |
| **Diversity & Serendipity** | Beyond-accuracy metrics: intra-list diversity and serendipity@K |
| **Explanation UI** | Each recommendation explains *why* it was suggested (genres, shared themes, user profile) |
| **Cold-Start Page** | New users pick genres → instant CB recommendations, no rating history needed |

---

## 📐 Algorithms

### Collaborative Filtering
- **User-Based CF**: Cosine similarity on mean-centred rating vectors, top-K neighbours
- **Item-Based CF**: Adjusted cosine similarity (mean-centred by user), Pearson-like
- **SVD** *(primary)*: Matrix factorisation with 100 latent factors, 20 epochs SGD

### Content-Based Filtering
- **TF-IDF** on combined `genres + user tags` (5000 features, unigrams + bigrams)
- User profile built as weighted average of rated movie TF-IDF vectors

### Hybrid Fusion (Dynamic)
```
w_cf = sigmoid(0.15 × (rating_count − 20))
w_cb = 1 − w_cf

score = w_cf × normalised_CF_score + w_cb × normalised_CB_score
```

### Evaluation Metrics
| Metric | Type |
|--------|------|
| RMSE, MAE | Prediction accuracy |
| Precision@K, Recall@K, F1@K, NDCG@K | Ranking quality |
| Coverage | Catalogue reach |
| Diversity@K | Intra-list variety |
| Serendipity@K | Unexpected + relevant |
| Novelty@K | Inverse popularity |

---

## 🧪 Running Evaluation Only (no UI)

```python
from src import load_and_prepare, UserCF, ItemCF, SVDModel, ContentBasedFilter
from src import evaluate_model, compare_models

data   = load_and_prepare("data")
ucf    = UserCF().fit(data["train_df"])
result = evaluate_model("UserCF", lambda u, n: ucf.recommend(u, n),
                         lambda u, m: ucf.predict(u, m),
                         data["train_df"], data["test_df"], data["movies"])
print(result)
```

---

## 📚 References
- Harper & Konstan (2015). *The MovieLens Datasets*. ACM TIIS.
- Koren, Bell & Volinsky (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
- Hug (2020). *Surprise: A Python library for recommender systems*. JOSS.

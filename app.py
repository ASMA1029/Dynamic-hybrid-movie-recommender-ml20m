"""
app.py — CineAI Hybrid Movie Recommender
Proper Netflix-style with search bar, user search, movie browsing
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import load_and_prepare
from src.collaborative import UserCF, ItemCF, SVDModel
from src.content_based import ContentBasedFilter
from src.hybrid import HybridRecommender
from src.explainer import Explainer
from src.evaluation import evaluate_model, compare_models

st.set_page_config(
    page_title="CineAI — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --red:    #E50914;
    --red2:   #FF1E2D;
    --black:  #000000;
    --dark:   #141414;
    --card:   #1F1F1F;
    --card2:  #2A2A2A;
    --border: #333333;
    --text:   #FFFFFF;
    --muted:  #8C8C8C;
    --grey:   #B3B3B3;
    --green:  #46d369;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--black) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.main .block-container { padding: 0 !important; max-width: 100% !important; background: var(--black) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0a0a0a !important; border-right: 1px solid #1a1a1a !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Buttons */
.stButton > button {
    background: var(--red) !important; color: white !important;
    border: none !important; border-radius: 4px !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    padding: 0.6rem 1.5rem !important; text-transform: uppercase !important;
    letter-spacing: 0.05em !important; transition: all 0.15s !important;
}
.stButton > button:hover { background: var(--red2) !important; box-shadow: 0 4px 15px rgba(229,9,20,0.4) !important; }

/* Inputs */
.stTextInput > div > div > input {
    background: #2a2a2a !important; border: 1px solid #444 !important;
    border-radius: 4px !important; color: white !important;
    font-size: 1rem !important; padding: 0.7rem 1rem !important;
}
.stTextInput > div > div > input:focus { border-color: white !important; }
.stSelectbox > div > div { background: #1F1F1F !important; border: 1px solid #333 !important; border-radius: 4px !important; }
.stSlider > div > div > div { background: var(--red) !important; }
.stMultiSelect > div > div { background: #1F1F1F !important; border: 1px solid #333 !important; border-radius: 4px !important; }

/* Expander */
details { background: #1a1a1a !important; border: 1px solid #2a2a2a !important; border-radius: 4px !important; margin-top: 0.3rem !important; }
summary { color: var(--grey) !important; font-size: 0.82rem !important; padding: 0.5rem 0.8rem !important; cursor: pointer !important; }

/* Metrics */
[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 4px !important; padding: 1rem !important; }
[data-testid="stMetricValue"] { color: var(--red) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--red); }

/* Dataframe */
.stDataFrame { border: 1px solid #333 !important; border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"

@st.cache_resource(show_spinner="🎬  Loading CineAI — Training models…")
def load_everything():
    data    = load_and_prepare(DATA_DIR)
    user_cf = UserCF(k=30, min_common=3)
    user_cf.fit(data["train_df"])
    item_cf = ItemCF(k=30)
    item_cf.fit(data["train_df"])
    svd = SVDModel(n_factors=100, n_epochs=20)
    svd.fit(data["train_df"])
    cb = ContentBasedFilter(max_features=5000, ngram_range=(1,2))
    cb.fit(data["movies"])
    all_ids = data["movies"]["movieId"].tolist()
    hybrid  = HybridRecommender(alpha=0.15, threshold=20, svd_weight=0.6)
    hybrid.attach(user_cf, item_cf, svd, cb, all_ids)
    explainer = Explainer(data["movies"], cb, data["train_df"])
    return {"data":data,"user_cf":user_cf,"item_cf":item_cf,
            "svd":svd,"cb":cb,"hybrid":hybrid,"explainer":explainer,"all_ids":all_ids}

# ── Helpers ──────────────────────────────────────────────────────
def get_title(movies_df, mid):
    row = movies_df[movies_df["movieId"]==mid]
    if row.empty: return f"Movie #{mid}"
    return row.iloc[0].get("title_clean", row.iloc[0]["title"])

def get_genres(movies_df, mid):
    row = movies_df[movies_df["movieId"]==mid]
    if row.empty: return []
    gl = row.iloc[0].get("genres_list",[])
    return list(gl) if gl else []

def genre_pills(genres):
    return "".join([f'<span style="display:inline-block;font-size:0.65rem;color:#888;border:1px solid #333;padding:0.1rem 0.4rem;border-radius:2px;margin:0.1rem 0.1rem 0 0;text-transform:uppercase;letter-spacing:0.05em">{g}</span>' for g in genres[:4]])

def match_color(pct):
    if pct >= 70: return "#46d369"
    if pct >= 40: return "#ffa500"
    return "#E50914"

def movie_card(rank, mid, score, movies_df, wcf=None, wcb=None, extra_label="Score"):
    t    = get_title(movies_df, mid)
    g    = get_genres(movies_df, mid)
    pct  = min(int(score * 100), 99)
    mc   = match_color(pct)
    cf_info = f'<span style="color:#555;font-size:0.72rem;margin-left:0.8rem">CF {wcf*100:.0f}% &middot; CB {wcb*100:.0f}%</span>' if wcf is not None else ""
    gp   = genre_pills(g)
    rank_str = f"{rank:02d}"
    html = (
        '<div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:4px;'
        'margin-bottom:0.5rem;overflow:hidden">' 
        '<div style="background:linear-gradient(135deg,#1a0505,#0d0d0d);'
        'padding:0.9rem 1.2rem 0.7rem;position:relative;min-height:80px">' 
        f'<span style="font-family:Oswald,sans-serif;font-size:3.5rem;font-weight:700;'
        f'color:rgba(255,255,255,0.05);position:absolute;right:0.5rem;bottom:-0.3rem">{rank_str}</span>'
        f'<div style="font-size:0.65rem;color:#E50914;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin-bottom:0.3rem">{extra_label}</div>'
        f'<div style="font-size:1rem;font-weight:600;color:#fff;line-height:1.3;margin-bottom:0.3rem">{t}</div>'
        f'<div>{gp}</div>'
        '</div>'
        f'<div style="padding:0.5rem 1.2rem 0.8rem;background:#1a1a1a">'
        f'<span style="color:{mc};font-size:0.82rem;font-weight:700">{pct}% Match</span>'
        f' {cf_info}'
        f'<div style="background:#2a2a2a;border-radius:1px;height:3px;margin-top:0.4rem">'
        f'<div style="width:{pct}%;height:100%;background:{mc};border-radius:1px"></div>'
        '</div></div></div>'
    )
    return html

# ── Sidebar ──────────────────────────────────────────────────────
def render_sidebar(movies_df, train_df):
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Oswald',sans-serif;font-size:1.8rem;font-weight:700;
                    color:#E50914;letter-spacing:0.05em;margin-bottom:0.1rem">🎬 CINEAI</div>
        <div style="font-size:0.68rem;color:#444;letter-spacing:0.15em;
                    text-transform:uppercase;margin-bottom:1.5rem">Hybrid AI Recommender</div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        page = st.radio("Navigation", [
            "🏠  Home",
            "🎯  For You",
            "🔍  Search Movies",
            "👤  User Lookup",
            "🆕  New User",
            "📊  Evaluation",
        ], label_visibility="collapsed")

        st.markdown("---")

        # Live stats
        n_users  = train_df["userId"].nunique()
        n_movies = movies_df["movieId"].nunique()
        n_ratings = len(train_df)
        st.markdown(f"""
        <div style="font-size:0.68rem;color:#E50914;text-transform:uppercase;
                    letter-spacing:0.1em;font-weight:600;margin-bottom:0.6rem">Live Stats</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-bottom:1rem">
            <div style="background:#1a1a1a;border:1px solid #222;border-radius:4px;padding:0.6rem;text-align:center">
                <div style="font-family:'Oswald',sans-serif;font-size:1.2rem;color:#E50914">{n_users:,}</div>
                <div style="font-size:0.6rem;color:#555;text-transform:uppercase">Users</div>
            </div>
            <div style="background:#1a1a1a;border:1px solid #222;border-radius:4px;padding:0.6rem;text-align:center">
                <div style="font-family:'Oswald',sans-serif;font-size:1.2rem;color:#E50914">{n_movies:,}</div>
                <div style="font-size:0.6rem;color:#555;text-transform:uppercase">Movies</div>
            </div>
        </div>
        <div style="background:#1a1a1a;border:1px solid #222;border-radius:4px;padding:0.6rem;text-align:center;margin-bottom:1rem">
            <div style="font-family:'Oswald',sans-serif;font-size:1.2rem;color:#E50914">{n_ratings:,}</div>
            <div style="font-size:0.6rem;color:#555;text-transform:uppercase">Training Ratings</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.68rem;color:#555;line-height:2.0">
        <div style="color:#E50914;font-weight:600;font-size:0.68rem;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem">AI Stack</div>
        SVD Matrix Factorisation<br>
        User-Based CF<br>
        Item-Based CF<br>
        TF-IDF Content Filter<br>
        Dynamic Sigmoid Fusion
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.68rem;color:#444;line-height:1.9">
        <div style="color:#E50914;font-weight:600;margin-bottom:0.2rem">Supervised By</div>
        <div style="color:#888">Dr. Shah Khalid</div>
        <div style="color:#555;margin-top:0.4rem;margin-bottom:0.2rem">IR Course Project</div>
        Asma Bibi<br>Nimra Hashmi<br>Samia Jamil
        </div>
        """, unsafe_allow_html=True)
    return page


# ══════════════════════════════════════════════
# PAGE 1 — HOME (Browse by genre)
# ══════════════════════════════════════════════
def page_home(objs):
    movies_df = objs["data"]["movies"]
    train_df  = objs["data"]["train_df"]
    cb        = objs["cb"]

    # ── Top Banner ──
    st.markdown("""
    <div style="background:linear-gradient(180deg,#1a0000 0%,#000 100%);
                padding:4rem 3rem 3rem;margin-bottom:2rem;position:relative;overflow:hidden">
        <div style="position:absolute;top:-100px;right:-100px;width:500px;height:500px;
                    background:radial-gradient(circle,rgba(229,9,20,0.12) 0%,transparent 70%);
                    border-radius:50%"></div>
        <div style="font-size:0.75rem;color:#E50914;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.2em;margin-bottom:0.5rem">Welcome to CineAI</div>
        <div style="font-family:'Oswald',sans-serif;font-size:4rem;font-weight:700;
                    color:#fff;line-height:1;margin-bottom:0.8rem;text-transform:uppercase">
            Discover Your<br><span style="color:#E50914">Next Favourite</span>
        </div>
        <div style="font-size:1rem;color:#888;max-width:500px;margin-bottom:1.5rem">
            AI-powered movie recommendations using SVD, Collaborative Filtering, and Content-Based Analysis
        </div>
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap">
            <div style="background:#E50914;color:white;font-size:0.75rem;font-weight:700;
                        padding:0.4rem 1rem;border-radius:2px;text-transform:uppercase">SVD Model</div>
            <div style="background:#1a1a1a;border:1px solid #333;color:#888;font-size:0.75rem;
                        padding:0.4rem 1rem;border-radius:2px;text-transform:uppercase">UserCF</div>
            <div style="background:#1a1a1a;border:1px solid #333;color:#888;font-size:0.75rem;
                        padding:0.4rem 1rem;border-radius:2px;text-transform:uppercase">ItemCF</div>
            <div style="background:#1a1a1a;border:1px solid #333;color:#888;font-size:0.75rem;
                        padding:0.4rem 1rem;border-radius:2px;text-transform:uppercase">TF-IDF CBF</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:0 3rem">', unsafe_allow_html=True)

    # ── Genre Browse ──
    all_genres = sorted({g for gl in movies_df["genres_list"]
                         for g in (gl if isinstance(gl,list) else [])})

    st.markdown("""
    <div style="font-family:'Oswald',sans-serif;font-size:1.3rem;font-weight:500;
                color:#fff;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:1rem">
    Browse by Genre
    </div>
    """, unsafe_allow_html=True)

    selected_genre = st.selectbox("Pick a genre", all_genres,
                                   index=all_genres.index("Action") if "Action" in all_genres else 0,
                                   label_visibility="collapsed")

    # Show top movies in that genre
    genre_movies = movies_df[movies_df["genres_list"].apply(
        lambda gl: selected_genre in (gl or [])
    )]["movieId"].tolist()

    # Rank by number of ratings
    rating_counts = train_df.groupby("movieId").size().reset_index(name="count")
    genre_rated   = rating_counts[rating_counts["movieId"].isin(genre_movies)]
    top_genre     = genre_rated.sort_values("count", ascending=False).head(20)

    st.markdown(f"""
    <div style="font-family:'Oswald',sans-serif;font-size:1.1rem;color:#E50914;
                text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.8rem">
    Top {selected_genre} Movies
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (_, row) in enumerate(top_genre.iterrows()):
        mid   = row["movieId"]
        t     = get_title(movies_df, mid)
        g     = get_genres(movies_df, mid)
        count = row["count"]
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:4px;
                        padding:0.8rem;margin-bottom:0.8rem;min-height:120px;
                        border-top:3px solid #E50914">
                <div style="font-size:0.6rem;color:#E50914;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem">
                    #{i+1} Most Rated
                </div>
                <div style="font-size:0.88rem;font-weight:600;color:#fff;
                            line-height:1.3;margin-bottom:0.4rem">{t}</div>
                <div style="margin-bottom:0.3rem">{genre_pills(g)}</div>
                <div style="font-size:0.72rem;color:#555">{count} ratings</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 2 — FOR YOU (Personalised Recommendations)
# ══════════════════════════════════════════════
def page_for_you(objs):
    data      = objs["data"]
    hybrid    = objs["hybrid"]
    explainer = objs["explainer"]
    movies_df = data["movies"]
    train_df  = data["train_df"]

    st.markdown('<div style="padding:2rem 3rem">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Oswald',sans-serif;font-size:2.5rem;font-weight:700;
                color:#fff;text-transform:uppercase;margin-bottom:0.3rem">For You</div>
    <div style="font-size:0.9rem;color:#888;margin-bottom:2rem">
    Personalised hybrid recommendations — SVD + UserCF + ItemCF + TF-IDF
    </div>
    """, unsafe_allow_html=True)

    # ── User Search ──
    st.markdown("""
    <div style="font-size:0.75rem;color:#E50914;font-weight:700;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:0.5rem">Enter Your User ID</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        user_input = st.text_input("User ID", placeholder="Type a User ID e.g. 1, 42, 100, 250...",
                                    label_visibility="collapsed")
    with col2:
        n_recs = st.slider("How many?", 5, 20, 10)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("▶  Get My Picks", use_container_width=True)

    # Show all available users hint
    all_users = sorted(train_df["userId"].unique())
    st.markdown(f'<div style="font-size:0.72rem;color:#444;margin-top:0.3rem">Available users: 1 to {max(all_users)} · Total: {len(all_users):,} users</div>', unsafe_allow_html=True)

    if go_btn:
        if not user_input.strip():
            st.warning("Please enter a User ID.")
        else:
            try:
                user_id = int(user_input.strip())
            except ValueError:
                st.error("User ID must be a number.")
                return

            if user_id not in all_users:
                st.error(f"User {user_id} not found. Try a number between {min(all_users)} and {max(all_users)}.")
                return

            with st.spinner(f"Finding perfect movies for User {user_id}…"):
                user_rated  = train_df[train_df["userId"]==user_id]
                rated_ids   = user_rated["movieId"].tolist()
                rated_ser   = user_rated.set_index("movieId")["rating"]
                weight_info = hybrid.explain_weights(user_id)
                recs        = hybrid.recommend(user_id, n=n_recs, rated_ids=rated_ids, ratings_series=rated_ser)
                profile     = explainer.profile_summary(user_id)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Profile strip ──
            n_rat   = profile.get("n_ratings", 0)
            avg_rat = profile.get("avg_rating", "—")
            w_cf    = weight_info["w_cf"]
            w_cb    = weight_info["w_cb"]
            prof    = weight_info["profile"]
            prof_color = {"cold-start":"#E50914","warming-up":"#ffa500","warm":"#46d369","experienced":"#00a8ff"}.get(prof,"#888")

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a0000,#1a1a1a);border:1px solid #2a2a2a;
                        border-radius:4px;padding:1.2rem 1.5rem;margin-bottom:1.5rem;
                        display:flex;align-items:center;gap:2rem;flex-wrap:wrap">
                <div style="background:#E50914;border-radius:4px;width:50px;height:50px;
                            display:flex;align-items:center;justify-content:center;
                            font-family:'Oswald',sans-serif;font-size:1.5rem;font-weight:700;flex-shrink:0">
                    {str(user_id)[0]}
                </div>
                <div>
                    <div style="font-size:0.68rem;color:#555;text-transform:uppercase;letter-spacing:0.1em">User</div>
                    <div style="font-family:'Oswald',sans-serif;font-size:1.5rem;color:#fff">#{user_id}</div>
                </div>
                <div style="width:1px;height:40px;background:#333"></div>
                <div>
                    <div style="font-size:0.68rem;color:#555;text-transform:uppercase">Ratings</div>
                    <div style="font-family:'Oswald',sans-serif;font-size:1.5rem;color:#E50914">{n_rat}</div>
                </div>
                <div>
                    <div style="font-size:0.68rem;color:#555;text-transform:uppercase">Avg Rating</div>
                    <div style="font-family:'Oswald',sans-serif;font-size:1.5rem;color:#fff">{avg_rat}</div>
                </div>
                <div style="width:1px;height:40px;background:#333"></div>
                <div>
                    <div style="font-size:0.68rem;color:#555;text-transform:uppercase;margin-bottom:0.3rem">AI Strategy</div>
                    <span style="background:{prof_color};color:#000;font-size:0.72rem;font-weight:700;
                                 padding:0.2rem 0.6rem;border-radius:2px;text-transform:uppercase">{prof}</span>
                </div>
                <div style="flex:1;min-width:200px">
                    <div style="font-size:0.68rem;color:#555;margin-bottom:0.3rem">
                        CF Weight <span style="color:#00a8ff">{w_cf*100:.0f}%</span>
                        &nbsp;·&nbsp;
                        CB Weight <span style="color:#E50914">{w_cb*100:.0f}%</span>
                    </div>
                    <div style="background:#2a2a2a;border-radius:2px;height:6px;overflow:hidden">
                        <div style="width:{w_cf*100:.0f}%;height:100%;background:#00a8ff;border-radius:2px"></div>
                    </div>
                    <div style="font-size:0.65rem;color:#444;margin-top:0.2rem">{weight_info["reason"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_recs, col_hist = st.columns([3, 1])

            with col_recs:
                st.markdown(f"""
                <div style="font-family:'Oswald',sans-serif;font-size:1.2rem;color:#fff;
                            text-transform:uppercase;letter-spacing:0.08em;margin-bottom:1rem">
                Top {n_recs} Picks For User #{user_id}
                </div>
                """, unsafe_allow_html=True)

                for rank, (mid, score, wcf, wcb) in enumerate(recs, 1):
                    st.markdown(movie_card(rank, mid, score, movies_df, wcf, wcb), unsafe_allow_html=True)
                    with st.expander(f"💬 Why #{rank}?"):
                        try:
                            exp = explainer.explain_hybrid(mid, user_id, wcf, wcb,
                                top_rated_ids=user_rated.nlargest(10,"rating")["movieId"].tolist())
                            st.markdown(exp)
                        except Exception:
                            st.markdown(f"**Collaborative:** {wcf*100:.0f}% · **Content:** {wcb*100:.0f}%")

            with col_hist:
                st.markdown("""
                <div style="font-family:'Oswald',sans-serif;font-size:1rem;color:#fff;
                            text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem">
                Watch History
                </div>
                """, unsafe_allow_html=True)

                top_movies = profile.get("top_movies", [])
                if top_movies:
                    for tm in top_movies[:8]:
                        stars = int(tm["rating"])
                        star_str = "★"*stars + "☆"*(5-stars)
                        st.markdown(f"""
                        <div style="padding:0.5rem 0;border-bottom:1px solid #1a1a1a">
                            <div style="font-size:0.8rem;color:#fff;overflow:hidden;
                                        white-space:nowrap;text-overflow:ellipsis">{tm["title"][:30]}</div>
                            <div style="font-size:0.72rem;color:#E50914">{star_str}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size:0.8rem;color:#555">No history available</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 3 — SEARCH MOVIES
# ══════════════════════════════════════════════
def page_search(objs):
    data      = objs["data"]
    cb        = objs["cb"]
    explainer = objs["explainer"]
    movies_df = data["movies"]
    train_df  = data["train_df"]

    st.markdown('<div style="padding:2rem 3rem">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Oswald',sans-serif;font-size:2.5rem;font-weight:700;
                color:#fff;text-transform:uppercase;margin-bottom:0.3rem">Search Movies</div>
    <div style="font-size:0.9rem;color:#888;margin-bottom:2rem">
    Search any title — see details and find similar movies
    </div>
    """, unsafe_allow_html=True)

    # ── Big Search Bar ──
    search_query = st.text_input("", placeholder="🔍  Search movie title e.g. 'Dark Knight', 'Inception', 'Toy Story'…",
                                  label_visibility="collapsed")

    title_map = {}
    for _, row in movies_df[["movieId","title_clean","title"]].iterrows():
        t = row.get("title_clean") or row.get("title","")
        if t: title_map[t.lower()] = (row["movieId"], t)

    if search_query.strip():
        query   = search_query.strip().lower()
        matches = [(mid, t) for key,(mid,t) in title_map.items() if query in key]
        matches = matches[:20]

        if not matches:
            st.markdown(f'<div style="color:#888;padding:2rem 0">No results for "<b style="color:white">{search_query}</b>"</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:0.8rem;color:#555;margin-bottom:1rem">{len(matches)} results for "{search_query}"</div>', unsafe_allow_html=True)

            selected_mid = None
            selected_title = None

            for mid, t in matches:
                g     = get_genres(movies_df, mid)
                count = len(train_df[train_df["movieId"]==mid])
                avg   = train_df[train_df["movieId"]==mid]["rating"].mean() if count > 0 else 0

                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:4px;
                                padding:0.9rem 1.2rem;margin-bottom:0.4rem">
                        <div style="font-size:1rem;font-weight:600;color:#fff;margin-bottom:0.3rem">{t}</div>
                        <div style="margin-bottom:0.3rem">{genre_pills(g)}</div>
                        <span style="font-size:0.72rem;color:#555">{count} ratings</span>
                        {"  ·  " + f'<span style="font-size:0.72rem;color:#46d369">★ {avg:.1f}</span>' if count > 0 else ""}
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("Similar →", key=f"sim_{mid}"):
                        selected_mid   = mid
                        selected_title = t

                if selected_mid:
                    st.markdown(f"""
                    <div style="font-family:'Oswald',sans-serif;font-size:1.1rem;color:#E50914;
                                text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.8rem">
                    Movies Similar to: {selected_title}
                    </div>
                    """, unsafe_allow_html=True)
                    similar = cb.recommend(selected_mid, n=10)
                    for rank, (smid, sscore) in enumerate(similar, 1):
                        st.markdown(movie_card(rank, smid, sscore, movies_df, extra_label="Similarity"), unsafe_allow_html=True)
                        with st.expander(f"💬 Shared features with {selected_title}"):
                            try:
                                st.markdown(explainer.explain_content(smid, selected_mid), unsafe_allow_html=True)
                            except Exception:
                                st.markdown(f"Content similarity: **{sscore:.3f}**")
                    break

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 4 — USER LOOKUP
# ══════════════════════════════════════════════
def page_user_lookup(objs):
    data      = objs["data"]
    movies_df = data["movies"]
    train_df  = data["train_df"]
    hybrid    = objs["hybrid"]
    svd       = objs["svd"]

    st.markdown('<div style="padding:2rem 3rem">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Oswald',sans-serif;font-size:2.5rem;font-weight:700;
                color:#fff;text-transform:uppercase;margin-bottom:0.3rem">User Lookup</div>
    <div style="font-size:0.9rem;color:#888;margin-bottom:2rem">
    Explore any user profile — their ratings, preferences, and AI weight breakdown
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_input("", placeholder="🔍  Enter User ID to explore their profile…",
                                label_visibility="collapsed")
    all_users  = sorted(train_df["userId"].unique())

    if user_input.strip():
        try:
            user_id = int(user_input.strip())
        except ValueError:
            st.error("Please enter a valid numeric User ID.")
            return

        if user_id not in all_users:
            st.error(f"User {user_id} not found.")
            return

        user_df   = train_df[train_df["userId"]==user_id]
        n_ratings = len(user_df)
        avg_rating= round(user_df["rating"].mean(), 2) if n_ratings > 0 else 0
        winfo     = hybrid.explain_weights(user_id)
        w_cf, w_cb = winfo["w_cf"], winfo["w_cb"]
        prof      = winfo["profile"]
        prof_color = {"cold-start":"#E50914","warming-up":"#ffa500","warm":"#46d369","experienced":"#00a8ff"}.get(prof,"#888")

        # Profile header
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a0000,#141414);border:1px solid #2a2a2a;
                    border-radius:4px;padding:2rem;margin-bottom:1.5rem">
            <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap">
                <div style="background:#E50914;border-radius:4px;width:70px;height:70px;
                            display:flex;align-items:center;justify-content:center;
                            font-family:'Oswald',sans-serif;font-size:2rem;font-weight:700">
                    {str(user_id)[0]}
                </div>
                <div>
                    <div style="font-size:0.68rem;color:#555;text-transform:uppercase">Profile</div>
                    <div style="font-family:'Oswald',sans-serif;font-size:2rem;color:#fff">User #{user_id}</div>
                    <span style="background:{prof_color};color:#000;font-size:0.72rem;font-weight:700;
                                 padding:0.2rem 0.6rem;border-radius:2px;text-transform:uppercase">{prof}</span>
                </div>
                <div style="margin-left:auto;text-align:right">
                    <div style="font-size:0.68rem;color:#555;margin-bottom:0.3rem">SIGMOID WEIGHTS</div>
                    <div style="font-size:1.5rem;color:#00a8ff;font-family:'Oswald',sans-serif">
                        CF {w_cf*100:.0f}%
                    </div>
                    <div style="font-size:1.5rem;color:#E50914;font-family:'Oswald',sans-serif">
                        CB {w_cb*100:.0f}%
                    </div>
                </div>
            </div>
            <div style="margin-top:1rem;background:#2a2a2a;border-radius:2px;height:8px;overflow:hidden">
                <div style="width:{w_cf*100:.0f}%;height:100%;background:linear-gradient(90deg,#00a8ff,#0060ff);border-radius:2px"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#555;margin-top:0.3rem">
                <span>Content-Based ←</span><span>→ Collaborative</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats
        col1,col2,col3,col4 = st.columns(4)
        for col, (num,label) in zip([col1,col2,col3,col4],[
            (n_ratings,"Total Ratings"),
            (avg_rating,"Avg Rating"),
            (f"{w_cf*100:.0f}%","CF Weight"),
            (f"{w_cb*100:.0f}%","CB Weight"),
        ]):
            col.markdown(f"""
            <div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:4px;
                        padding:1rem;text-align:center;margin-bottom:1rem">
                <div style="font-family:'Oswald',sans-serif;font-size:1.8rem;color:#E50914">{num}</div>
                <div style="font-size:0.65rem;color:#555;text-transform:uppercase;letter-spacing:0.1em">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        # Rating history table
        st.markdown("""
        <div style="font-family:'Oswald',sans-serif;font-size:1.1rem;color:#fff;
                    text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.8rem">
        Rating History
        </div>
        """, unsafe_allow_html=True)

        history = user_df.merge(movies_df[["movieId","title_clean","genres_list"]], on="movieId", how="left")
        history = history.sort_values("rating", ascending=False)
        for _, row in history.head(15).iterrows():
            t     = row.get("title_clean") or f"Movie #{row['movieId']}"
            r     = row["rating"]
            stars = "★" * int(r) + "☆" * (5 - int(r))
            g     = row.get("genres_list",[])
            color = "#46d369" if r >= 4 else "#ffa500" if r >= 3 else "#E50914"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1rem;padding:0.5rem 0;border-bottom:1px solid #1a1a1a">
                <div style="font-family:'Oswald',sans-serif;font-size:1.5rem;font-weight:700;
                            color:{color};width:40px;text-align:right;flex-shrink:0">{r}</div>
                <div style="flex:1">
                    <div style="font-size:0.88rem;color:#fff">{t}</div>
                    <div style="font-size:0.65rem;color:{color}">{stars}</div>
                </div>
                <div>{genre_pills(list(g) if g else [])}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 5 — NEW USER (Cold Start)
# ══════════════════════════════════════════════
def page_new_user(objs):
    data      = objs["data"]
    cb        = objs["cb"]
    explainer = objs["explainer"]
    movies_df = data["movies"]

    st.markdown('<div style="padding:2rem 3rem">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Oswald',sans-serif;font-size:2.5rem;font-weight:700;
                color:#fff;text-transform:uppercase;margin-bottom:0.3rem">New User</div>
    <div style="font-size:0.9rem;color:#888;margin-bottom:0.5rem">
    No account? No problem. Pick your genres and get instant AI recommendations.
    </div>
    <div style="background:#1a0000;border:1px solid #2a0000;border-radius:4px;
                padding:0.8rem 1.2rem;margin-bottom:2rem;font-size:0.82rem;color:#888">
    <span style="color:#E50914;font-weight:700">Cold-Start Mode</span> — 
    Our sigmoid fusion gives 95% weight to Content-Based Filtering when no rating history exists.
    As you rate movies, collaborative filtering gradually takes over.
    </div>
    """, unsafe_allow_html=True)

    all_genres = sorted({g for gl in movies_df["genres_list"]
                         for g in (gl if isinstance(gl,list) else [])})

    col1, col2 = st.columns([3,1])
    with col1:
        selected = st.multiselect("Genres you love", all_genres,
                                   default=["Action","Thriller"],
                                   label_visibility="collapsed",
                                   placeholder="Pick genres you enjoy…")
    with col2:
        n = st.slider("Results", 5, 20, 10)

    if st.button("▶  Show My Movies", use_container_width=False):
        if not selected:
            st.warning("Select at least one genre.")
        else:
            from sklearn.metrics.pairwise import cosine_similarity as cs
            query_vec  = cb._tfidf.transform([" ".join(selected)])
            genre_mask = movies_df["genres_list"].apply(lambda gl: any(g in (gl or []) for g in selected))
            valid_ids  = [m for m in movies_df[genre_mask]["movieId"].tolist() if m in cb._id2idx]

            if not valid_ids:
                st.warning("No movies found.")
            else:
                target_mat = cb._tfidf_mat[[cb._id2idx[m] for m in valid_ids]]
                sims       = cs(query_vec, target_mat).flatten()
                scored     = sorted(zip(valid_ids, sims.tolist()), key=lambda x:x[1], reverse=True)

                # Genre badges
                badges = "".join([f'<span style="background:#E50914;color:white;font-size:0.72rem;font-weight:700;padding:0.3rem 0.8rem;border-radius:2px;margin-right:0.3rem;text-transform:uppercase">{g}</span>' for g in selected])
                st.markdown(f'<div style="margin:1rem 0;display:flex;flex-wrap:wrap;gap:0.3rem;align-items:center">{badges}<span style="color:#555;font-size:0.78rem">{len(valid_ids)} movies found</span></div>', unsafe_allow_html=True)

                st.markdown("""
                <div style="font-family:'Oswald',sans-serif;font-size:1.1rem;color:#fff;
                            text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem">
                Your Personalised List
                </div>
                """, unsafe_allow_html=True)

                for rank, (mid, score) in enumerate(scored[:n], 1):
                    st.markdown(movie_card(rank, mid, score, movies_df, extra_label="Genre Match"), unsafe_allow_html=True)
                    with st.expander(f"💬 Why #{rank}?"):
                        try:
                            st.markdown(explainer.explain_cold_start(mid, selected), unsafe_allow_html=True)
                        except Exception:
                            st.markdown(f"Matched your interest in: **{', '.join(selected)}**")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 6 — EVALUATION
# ══════════════════════════════════════════════
def page_evaluation(objs):
    data = objs["data"]; hybrid = objs["hybrid"]; cb = objs["cb"]
    user_cf=objs["user_cf"]; item_cf=objs["item_cf"]; svd=objs["svd"]
    train_df=data["train_df"]; test_df=data["test_df"]; movies_df=data["movies"]

    st.markdown('<div style="padding:2rem 3rem">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Oswald',sans-serif;font-size:2.5rem;font-weight:700;
                color:#fff;text-transform:uppercase;margin-bottom:0.3rem">Model Dashboard</div>
    <div style="font-size:0.9rem;color:#888;margin-bottom:2rem">
    Precision · Recall · NDCG · Diversity · Serendipity · Novelty
    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3,col4 = st.columns(4)
    for col,(num,label) in zip([col1,col2,col3,col4],[
        (f"{len(train_df):,}","Training Ratings"),
        (f"{train_df['userId'].nunique():,}","Users"),
        (f"{train_df['movieId'].nunique():,}","Movies"),
        ("5","AI Models"),
    ]):
        col.markdown(f'<div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:4px;padding:1rem;text-align:center;margin-bottom:1rem"><div style="font-family:\'Oswald\',sans-serif;font-size:1.8rem;color:#E50914">{num}</div><div style="font-size:0.65rem;color:#555;text-transform:uppercase;letter-spacing:0.1em">{label}</div></div>', unsafe_allow_html=True)

    col_k,col_n = st.columns(2)
    with col_k: k = st.slider("K (cutoff)", 5, 20, 10)
    with col_n: n_eval = st.slider("Users to evaluate", 20, 100, 50)

    st.markdown('<div style="background:#1a0000;border:1px solid #2a0000;border-radius:4px;padding:0.8rem 1.2rem;font-size:0.82rem;color:#888;margin:0.5rem 0">⚡ Runs all 5 models — approximately 2 minutes. Cached after first run.</div>', unsafe_allow_html=True)

    if st.button("▶  Run Full Evaluation", use_container_width=False):
        with st.spinner("Evaluating all models…"):
            def sim_fn(a,b): return cb.similarity_between(a,b)
            results=[]
            results.append(evaluate_model("UserCF",lambda uid,n:user_cf.recommend(uid,n),lambda uid,mid:user_cf.predict(uid,mid),train_df,test_df,movies_df,similarity_fn=sim_fn,k=k,max_users=n_eval))
            results.append(evaluate_model("ItemCF",lambda uid,n:item_cf.recommend(uid,n),lambda uid,mid:item_cf.predict(uid,mid),train_df,test_df,movies_df,similarity_fn=sim_fn,k=k,max_users=n_eval))
            results.append(evaluate_model("SVD",lambda uid,n:svd.recommend(uid,n),lambda uid,mid:svd.predict(uid,mid),train_df,test_df,movies_df,similarity_fn=sim_fn,k=k,max_users=n_eval))
            def cb_rec(uid,n):
                ur=train_df[train_df["userId"]==uid]
                return cb.recommend_for_user(ur["movieId"].tolist(),ur.set_index("movieId")["rating"],n=n)
            results.append(evaluate_model("ContentBased",cb_rec,None,train_df,test_df,movies_df,similarity_fn=sim_fn,k=k,max_users=n_eval))
            def hyb_rec(uid,n):
                ur=train_df[train_df["userId"]==uid]
                recs=hybrid.recommend(uid,n=n,rated_ids=ur["movieId"].tolist(),ratings_series=ur.set_index("movieId")["rating"])
                return [(m,s) for m,s,*_ in recs]
            results.append(evaluate_model("Hybrid",hyb_rec,None,train_df,test_df,movies_df,similarity_fn=sim_fn,k=k,max_users=n_eval))
            st.session_state["eval_results"] = compare_models(results)

    if "eval_results" in st.session_state:
        df_results = st.session_state["eval_results"]
        LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#B3B3B3",family="Inter"),
                      legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor="#333"),
                      xaxis=dict(gridcolor="#1a1a1a"),yaxis=dict(gridcolor="#1a1a1a"))
        COLORS = ["#E50914","#B3B3B3","#46d369","#ffa500","#00a8ff"]

        st.markdown('<div style="font-family:\'Oswald\',sans-serif;font-size:1.1rem;color:#fff;text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.8rem">Results Table</div>', unsafe_allow_html=True)
        st.dataframe(df_results, use_container_width=True)

        numeric_cols   = [c for c in df_results.columns if df_results[c].dtype in [np.float64,np.int64]]
        rank_metrics   = [c for c in numeric_cols if any(x in c for x in ["P@","R@","F1","NDCG","Coverage"])]
        beyond_metrics = [c for c in numeric_cols if any(x in c for x in ["Diversity","Serendipity","Novelty"])]

        col1,col2 = st.columns(2)
        with col1:
            if rank_metrics:
                fig = px.bar(df_results[rank_metrics].reset_index().melt(id_vars="Model"),
                             x="variable",y="value",color="Model",barmode="group",
                             title=f"Ranking Metrics @{k}",color_discrete_sequence=COLORS)
                fig.update_layout(**LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if beyond_metrics:
                fig = px.bar(df_results[beyond_metrics].reset_index().melt(id_vars="Model"),
                             x="variable",y="value",color="Model",barmode="group",
                             title=f"Beyond-Accuracy @{k}",color_discrete_sequence=COLORS)
                fig.update_layout(**LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        # Sigmoid scatter
        st.markdown('<div style="font-family:\'Oswald\',sans-serif;font-size:1.1rem;color:#fff;text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.8rem">Sigmoid Weight Distribution</div>', unsafe_allow_html=True)
        wd=[]
        for uid in train_df["userId"].unique()[:200]:
            wi=hybrid.explain_weights(uid)
            wd.append({"userId":uid,"rating_count":svd.rating_count(uid),"w_cf":wi["w_cf"],"profile":wi["profile"]})
        wdf=pd.DataFrame(wd)
        fig=px.scatter(wdf,x="rating_count",y="w_cf",color="profile",
                       title="CF Weight vs Rating Count — Dynamic Cold-Start Sigmoid",
                       labels={"rating_count":"Ratings","w_cf":"CF Weight"},
                       color_discrete_map={"cold-start":"#E50914","warming-up":"#ffa500","warm":"#46d369","experienced":"#00a8ff"})
        fig.add_hline(y=0.5,line_dash="dash",line_color="#555",annotation_text="θ=20 Equal Split")
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    try:
        objs = load_everything()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Place ratings.csv, movies.csv, tags.csv in a data/ folder.")
        return

    page = render_sidebar(objs["data"]["movies"], objs["data"]["train_df"])

    if   "Home"         in page: page_home(objs)
    elif "For You"      in page: page_for_you(objs)
    elif "Search"       in page: page_search(objs)
    elif "User Lookup"  in page: page_user_lookup(objs)
    elif "New User"     in page: page_new_user(objs)
    elif "Evaluation"   in page: page_evaluation(objs)

if __name__ == "__main__":
    main()

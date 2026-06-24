"""
Movie Recommender - Streamlit app
Finds movies similar to one you like, using latent factors learned by SVD
matrix factorization (no popularity bias from naive zero-filling).

Deployed on Streamlit Community Cloud. The model retrains on startup from
the two CSVs (~seconds), so there's no large model file to host on GitHub.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from surprise import Dataset, Reader, SVD
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


# ---------- helpers ----------
def fix_title(title):
    """MovieLens stores 'Matrix, The (1999)'; flip it to 'The Matrix (1999)'."""
    m = re.match(r'^(.+), (The|A|An) (\(\d{4}\))$', title)
    if m:
        return f"{m.group(2)} {m.group(1)} {m.group(3)}"
    return title


# ---------- load + train (cached so it runs only once) ----------
@st.cache_resource(show_spinner="Training the recommender (one-time, ~10s)...")
def build_model():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # train SVD on ALL ratings (full trainset - we're deploying, not evaluating here)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=100, random_state=42)
    model.fit(trainset)

    # movie latent-factor matrix, and the map from matrix position -> raw movieId
    item_factors = model.qi
    raw_movieids = [trainset.to_raw_iid(i) for i in range(item_factors.shape[0])]

    # similarity computed on LEARNED factors -> no popularity bias
    item_sim = cosine_similarity(item_factors)

    movieid_to_pos = {raw: pos for pos, raw in enumerate(raw_movieids)}
    movieid_to_title = dict(zip(movies['movieId'], movies['title']))

    # only offer movies that actually have a learned factor (i.e. were rated)
    rated_ids = set(raw_movieids)
    selectable = movies[movies['movieId'].isin(rated_ids)].copy()
    selectable['display'] = selectable['title'].apply(fix_title)
    selectable = selectable.sort_values('display')

    return item_sim, raw_movieids, movieid_to_pos, movieid_to_title, selectable


def recommend_similar(title_to_mid, item_sim, raw_movieids,
                      movieid_to_pos, movieid_to_title, mid, n=10):
    pos = movieid_to_pos[mid]
    scores = list(enumerate(item_sim[pos]))
    scores.sort(key=lambda x: x[1], reverse=True)
    scores = scores[1:n + 1]  # skip the movie itself (position 0, score 1.0)
    rows = []
    for p, s in scores:
        raw = raw_movieids[p]
        rows.append({
            'Movie': fix_title(movieid_to_title[raw]),
            'Similarity': round(float(s), 3)
        })
    return pd.DataFrame(rows)


# ---------- UI ----------
st.title("🎬 Movie Recommender")
st.markdown(
    "Pick a movie you like - get recommendations based on **latent taste factors** "
    "learned by SVD matrix factorization from 100,000+ real ratings. "
    "Unlike a genre-matching system, this captures *what kind of people like what*."
)

item_sim, raw_movieids, movieid_to_pos, movieid_to_title, selectable = build_model()
title_to_mid = dict(zip(selectable['display'], selectable['movieId']))

choice = st.selectbox(
    "Choose a movie:",
    options=selectable['display'].tolist(),
    index=None,
    placeholder="Start typing a title...",
)

n = st.slider("How many recommendations?", 5, 20, 10)

if choice:
    mid = title_to_mid[choice]
    recs = recommend_similar(title_to_mid, item_sim, raw_movieids,
                             movieid_to_pos, movieid_to_title, mid, n=n)
    st.subheader(f"Because you liked *{choice}*:")
    st.dataframe(recs, use_container_width=True, hide_index=True)
    st.caption(
        "Similarity is cosine distance between movies in 100-dimensional "
        "latent-factor space. Higher = more similar taste profile."
    )

st.divider()
st.markdown(
    "<small>Built with SVD (scikit-surprise). Part of a data science portfolio - "
    "[see the full project & two other methods on GitHub](https://github.com/sigunthedeer).</small>",
    unsafe_allow_html=True,
)

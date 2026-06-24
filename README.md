# 🎬 Movie Recommender System - MovieLens

A movie recommendation engine built three different ways - content-based filtering, item-item collaborative filtering, and matrix factorization (SVD) - with an honest comparison of what each method does well and where it breaks. Includes a live Streamlit app.

**[🚀 Live Demo](https://movie-recommender-uuuymejwgjrn3ejcbttnk7.streamlit.app/)** &nbsp;·&nbsp; built on the [MovieLens small dataset](https://grouplens.org/datasets/movielens/) (100,836 ratings · 610 users · 9,724 movies)

---

## TL;DR

I built the same recommender three ways, each fixing a flaw in the previous one:

| Method | Idea | Strength | Weakness |
|---|---|---|---|
| **1. Content-based** | Movies similar by **genre** (cosine similarity on genre vectors) | Simple, no rating data needed, no cold-start | Only knows genres - can't tell *Pulp Fiction* from any other crime film |
| **2. Item-item CF** | Movies rated similarly by the **same people** | Captures taste genres can't (links *Toy Story* → *Star Wars*) | Popularity bias - recommends blockbusters regardless of seed |
| **3. Matrix factorization (SVD)** | Learns hidden **latent factors** from real ratings | Personalized per-user, no popularity bias, **measurable** | More complex, needs enough ratings per user |

**Headline result:** the SVD model predicts held-out ratings with **RMSE 0.8807 / MAE 0.6766** on a 0.5–5.0 scale - i.e. predictions land within ~0.88 stars of the truth, on data the model never trained on.

---

## Why three methods?

Most recommender tutorials stop at one approach. I built all three on purpose, because the interesting part is *why you'd move from one to the next*:

1. **Content-based filtering** is the obvious start - but feeding it *Toy Story* returns five animated kids' films, and feeding it *Pulp Fiction* returns a pile of crime-thrillers it can't meaningfully rank. The lesson: **genres are coarse features.** The model is only as smart as what you describe movies with.

2. **Collaborative filtering** fixes that by ignoring genres entirely and asking *"what did people who liked this also like?"* Feeding it *Toy Story* surfaces *Star Wars*, *Forrest Gump*, *Jurassic Park* - connections no genre system could find. But it has a new flaw: because unrated movies are filled with 0, popular films correlate with everything, so recommendations drift toward blockbusters.

3. **Matrix factorization** fixes *that* by learning from only the ratings that exist - no zero-filling lie. It discovers hidden "taste dimensions" (latent factors) and predicts how any user would rate any movie. The recommendations become genuinely personal: user 1 in the dataset gets *Blade Runner*, *Seven Samurai*, *Casablanca* - a coherent film-buff profile the model inferred purely from rating patterns.

Each method's weakness is the next method's reason to exist. That progression is the whole project.

---

## Key concepts demonstrated

- **One-hot encoding** of multi-valued categorical data (`str.get_dummies('|')`)
- **Cosine similarity** for both genre vectors and rating vectors
- **Sparsity** - quantified the user-item matrix at **98.30% empty**, the core challenge of CF
- **Popularity bias** - identified, explained, and then *fixed* via factorization
- **Matrix factorization / SVD** with latent factors (the Netflix Prize approach)
- **Honest evaluation** - train/test split + RMSE/MAE on held-out ratings
- **Two recommendation modes** - item-centric ("movies like X") and user-centric ("what should user U watch?")

---

## Results in detail

### Content-based - *Toy Story (1995)*
Returns: *Antz*, *Toy Story 2*, *Adventures of Rocky and Bullwinkle*, *The Emperor's New Groove*, *Monsters Inc.*
→ A tight animated-family cluster. Sensible but shallow - it only sees shared genre tags.

### Item-item CF - *Toy Story (1995)*
Returns: *Star Wars*, *Forrest Gump*, *Jurassic Park*, *Independence Day*, *Toy Story 2*
→ Cross-genre connections genres can't make - but the popular-blockbuster lean is visible.

### Matrix factorization - *user 1*
Returns: *Blade Runner*, *Dr. Strangelove*, *North by Northwest*, *Casablanca*, *Seven Samurai*, *The Departed*…
→ A coherent, personalized taste profile. No popularity padding. **RMSE 0.8807.**

---

## Tech stack

`pandas` · `scikit-learn` (cosine similarity) · `scikit-surprise` (SVD) · `Streamlit` (deployment)

---

## Honest limitations

- **Cold start:** a brand-new user with no ratings can't be served by the SVD model - there's nothing to learn their taste from. A real system would fall back to content-based or popularity recommendations until enough ratings accrue.
- **Small dataset:** 610 users is modest. More users/ratings would sharpen the latent factors.
- **No temporal modeling:** I ignored the `timestamp` column. Taste drifts over time; a production system might weight recent ratings more.
- **Evaluation measures rating prediction, not ranking quality:** RMSE tells us how close predicted ratings are, not whether the *top-N list* is well-ordered. Metrics like precision@k would test the actual recommendations.

---

## Repository structure

```
movie-recommender/
├── README.md
├── requirements.txt
├── streamlit_app.py          # the deployed app (retrains SVD on startup)
├── movies.csv                # MovieLens movie metadata
├── ratings.csv               # MovieLens ratings
└── notebook.ipynb            # full build: all three methods, step by step
```

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

*Project 7 of my data science portfolio. Built as part of a self-directed transition into data science - focused on understanding methods deeply enough to explain their tradeoffs, not just running them.*

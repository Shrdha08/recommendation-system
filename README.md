# Movie Recommendation System

A full-stack movie recommendation platform built on the MovieLens 20M dataset.
Four recommendation approaches — from classical collaborative filtering to deep
learning — were implemented and benchmarked against each other; the Neural
Collaborative Filtering (NeuMF) model is served through a REST API with a React
frontend.

**Stack:** Python · TensorFlow/Keras · NumPy · FastAPI · PostgreSQL · React · Docker

---

## Quick Start

The entire stack runs with a single command:

```bash
docker compose up
```

Then open **http://localhost:5173**.

That builds the API, the frontend, and PostgreSQL, seeds the database, and
serves the app. No local Python, Node, or database setup is required — only
Docker.

> **First run:** seeding loads 404k ratings and takes roughly 30 seconds. Until
> it completes the page will report that the API is unreachable; it works as
> soon as the backend logs `Application startup complete`. Subsequent starts
> skip seeding and come up immediately.

| Service | URL | Notes |
|---|---|---|
| Frontend | http://localhost:5173 | nginx; proxies `/api` to the backend |
| API | http://localhost:8000 | FastAPI |
| API docs | http://localhost:8000/docs | Auto-generated OpenAPI UI |

Useful variations:

```bash
docker compose up -d          # run detached
docker compose logs -f        # follow logs
docker compose down           # stop
docker compose down -v        # stop and delete database volume
```

Seeding is idempotent — restarts reuse the existing data rather than reloading it.

---

## Architecture

```
                  ┌──────────────┐
   Browser ──────▶│   frontend   │  React (Vite) build served by nginx
                  │  :5173 → :80 │  proxies /api/* ──┐
                  └──────────────┘                   │
                                                     ▼
                  ┌──────────────┐          ┌──────────────┐
                  │      db      │◀────────▶│   backend    │
                  │  PostgreSQL  │          │   FastAPI    │
                  │    :5432     │          │    :8000     │
                  └──────────────┘          └──────┬───────┘
                                                   │
                                          NeuMF weights (NumPy)
```

PostgreSQL stores the movie catalogue and rating history. The backend loads the
trained NeuMF weights once at startup and scores candidate movies in-process,
excluding titles the user has already rated. Because nginx proxies `/api`, the
application is single-origin in Docker.

### Repository layout

```
├── docker-compose.yml           # full stack: db + backend + frontend
├── src/                         # data preprocessing and model preparation
│   ├── preprocess.py            # raw MovieLens → sequential ID mapping
│   ├── preprocess_shrink.py     # reduce to top 10k users / 2k movies
│   ├── preprocess_sparse.py     # sparse matrix construction
│   ├── preprocess2dict.py       # dictionary lookups for CF baseline
│   ├── reconstruct_mappings.py  # rebuild ID → title mapping (no retraining)
│   └── extract_weights.py       # export Keras weights → NumPy .npz
├── models/
│   ├── collaborative filtering/
│   ├── matrix factorization/
│   └── neural collaborative filtering/
├── backend/                     # FastAPI service
│   ├── app/
│   │   ├── main.py              # routes
│   │   ├── ncf_model.py         # NeuMF inference (NumPy)
│   │   ├── models.py            # SQLAlchemy ORM
│   │   ├── schemas.py           # Pydantic schemas
│   │   └── database.py
│   └── seed.py                  # idempotent database seeding
└── frontend/                    # React + Vite client
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health check |
| `GET` | `/api/summary` | User and movie counts |
| `GET` | `/api/movies` | Browse/search catalogue (`search`, `limit`, `offset`) |
| `GET` | `/api/users/{user_id}/profile` | A user's highest-rated movies (`limit`) |
| `GET` | `/api/users/{user_id}/recommendations` | NeuMF predictions (`top_k`) |

Example:

```bash
curl "http://localhost:8000/api/users/42/recommendations?top_k=5"
```

```json
[
  {
    "movie_idx": 937,
    "title": "Indecent Proposal (1993)",
    "genres": "Drama|Romance",
    "predicted_rating": 3.486
  }
]
```

Valid user IDs are `0–3373`; out-of-range values return `404`.

---

## Dataset

The raw MovieLens 20M files are not committed due to size. To regenerate the
processed dataset from scratch:

1. Download the [MovieLens 20M dataset](https://grouplens.org/datasets/movielens/20m/)
2. Place `rating.csv` and `movie.csv` in `data/raw/`
3. Run the preprocessing scripts in `src/`

Processed output is written to `data/processed/`. The two files the application
needs at runtime (`final_movies.csv`, `final_ratings.csv`) are committed, so
`docker compose up` works from a fresh clone without this step.

The training subset covers **3,374 users**, **2,000 movies**, and **404,128 ratings**,
obtained by taking the most active users and most-rated movies from the first
500k rating records.

---

## Recommendation Models

Four approaches were implemented and evaluated on the same train/test split.

### 1. User-Based Collaborative Filtering (baseline)

* Cosine similarity between users
* Ratings predicted from the preferences of similar users

### 2. Matrix Factorization (ALS)

Decomposes the user–item interaction matrix into latent factors with bias terms,
optimised by Alternating Least Squares:

> r̂(u, i) = μ + bᵤ + bᵢ + pᵤᵀqᵢ

Alternately fixes user factors to solve for item factors, and vice versa.

### 3. Neural Matrix Factorization (Keras)

Learns user and movie embeddings by gradient descent rather than ALS:

> r̂(u, i) = Uᵤ · Wᵢ + bᵤ + bᵢ

### 4. Neural Collaborative Filtering — NeuMF (served model)

Combines a **Generalized Matrix Factorization** branch, which captures linear
interactions, with a **Multi-Layer Perceptron** branch that captures non-linear
ones:

```
     userId                    movieId
        │                         │
   ┌────┴─────┬─────────────┬─────┴────┐
   │          │             │          │
 GMF user  MLP user     MLP movie   GMF movie
 embedding embedding    embedding   embedding
   │          └──────┬──────┘          │
   │            concatenate            │
   │                 │                 │
   │           Dense(64) ReLU          │
   │            Dropout(0.2)           │
   │           Dense(32) ReLU          │
   │            Dropout(0.2)           │
   └────────┐        │                 │
        elementwise multiply ◀─────────┘
            │        │
          concatenate(GMF, MLP)
                 │
             Dense(1) → r̂
```

Trained with MSE loss, L2 regularization, dropout, and early stopping.

### Results

| Model | Test MSE | RMSE |
|---|---|---|
| User-Based Collaborative Filtering | 0.666 | 0.82 |
| **Matrix Factorization (ALS)** | **0.540** | **0.73** |
| Neural Matrix Factorization | 0.568 | 0.75 |
| Neural Collaborative Filtering (NeuMF) | 0.685 | 0.83 |

Matrix Factorization with ALS gave the strongest results, improving RMSE by
~11% over the collaborative filtering baseline.

---

## Implementation Notes

### Serving NeuMF without TensorFlow

The API does not depend on TensorFlow. `src/extract_weights.py` reads the
architecture graph from the saved model's `config.json` and its weight arrays
from `model.weights.h5`, exporting them to a NumPy `.npz`. The forward pass is
reimplemented in `backend/app/ncf_model.py` using the layer wiring taken from
the model's own graph, making it numerically equivalent to the Keras model while
keeping the runtime image small and removing a heavyweight dependency from the
serving path.

### Reconstructing the ID → title mapping

Model training used remapped, contiguous user and movie indices, and the lookup
tables were not committed. `src/reconstruct_mappings.py` recovers them by
replaying the same deterministic preprocessing steps, then asserts that the
resulting dimensions match the saved model's embedding shapes (N=3374, M=2000)
— so the mapping is verified against the checkpoint rather than assumed.

### Known limitation of the current checkpoint

In the saved `ncf_model.keras`, the MLP-branch embeddings have a standard
deviation of ~0.001, compared to ~0.07 for the GMF branch — they remained close
to their initialisation because early stopping (`patience=2`) halted training
before that branch learned meaningful representations. Consequently the served
predictions vary only slightly across users and cluster near the global mean
rating (~3.42–3.49), consistent with NeuMF's weaker test MSE above.

Retraining with increased patience, a warm-up schedule, or a higher learning
rate for the MLP branch would address this. The serving path is independent of
the fix and would pick up a new checkpoint by re-running `extract_weights.py`.

---

## Local Development

To run the services directly instead of through Docker:

```bash
# database
docker compose up db -d

# backend  (http://localhost:8000)
cd backend
pip install -r requirements.txt
python3 seed.py                  # reads data from ../data/processed
uvicorn app.main:app --reload

# frontend  (http://localhost:5173)
cd frontend
npm install
npm run dev
```

The backend defaults to the containerised database on `localhost:5433`; override
with `DATABASE_URL`. The frontend defaults to `http://localhost:8000` for the
API; override with `VITE_API_URL`.

**Note:** the backend requires Python ≤ 3.13 for the `psycopg` binary wheel.
The Docker image pins Python 3.12, which is the recommended path.

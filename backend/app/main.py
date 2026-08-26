from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from .database import get_db
from .models import Movie, Rating
from .ncf_model import get_model
from .schemas import MovieOut, RatedMovieOut, RecommendationOut, UserSummaryOut

app = FastAPI(title="Movie Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/summary", response_model=UserSummaryOut)
def summary():
    model = get_model()
    return UserSummaryOut(user_count=model.n_users, movie_count=model.n_movies)


@app.get("/api/movies", response_model=list[MovieOut])
def list_movies(
    search: str | None = None,
    limit: int = Query(30, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(Movie)
    if search:
        q = q.filter(Movie.title.ilike(f"%{search}%"))
    return q.order_by(Movie.title).offset(offset).limit(limit).all()


def _validate_user(user_id: int):
    model = get_model()
    if not (0 <= user_id < model.n_users):
        raise HTTPException(status_code=404, detail=f"user_id must be in [0, {model.n_users - 1}]")


@app.get("/api/users/{user_id}/profile", response_model=list[RatedMovieOut])
def user_profile(user_id: int, limit: int = 10, db: Session = Depends(get_db)):
    _validate_user(user_id)
    rows = (
        db.query(Movie, Rating.rating)
        .join(Rating, Rating.movie_idx == Movie.movie_idx)
        .filter(Rating.user_idx == user_id)
        .order_by(Rating.rating.desc())
        .limit(limit)
        .all()
    )
    return [RatedMovieOut(movie_idx=m.movie_idx, title=m.title, genres=m.genres, rating=r) for m, r in rows]


@app.get("/api/users/{user_id}/recommendations", response_model=list[RecommendationOut])
def user_recommendations(user_id: int, top_k: int = 10, db: Session = Depends(get_db)):
    _validate_user(user_id)
    model = get_model()

    rated_movie_idxs = {
        row[0]
        for row in db.query(Rating.movie_idx).filter(Rating.user_idx == user_id).all()
    }

    recs = model.recommend_for_user(user_id, rated_movie_idxs, top_k=top_k)
    movie_idxs = [m for m, _ in recs]
    movies = {m.movie_idx: m for m in db.query(Movie).filter(Movie.movie_idx.in_(movie_idxs)).all()}

    return [
        RecommendationOut(
            movie_idx=idx,
            title=movies[idx].title,
            genres=movies[idx].genres,
            predicted_rating=round(score, 3),
        )
        for idx, score in recs
        if idx in movies
    ]

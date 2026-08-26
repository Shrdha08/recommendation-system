"""Seeds Postgres with the reconstructed movie catalog and ratings.

Idempotent: by default it skips work if the database already holds data, so it
is safe to run on every container start. Pass --force to rebuild from scratch.
"""
import argparse
import os
import sys

import pandas as pd
from sqlalchemy import func, text

from app.database import Base, SessionLocal, engine
from app.models import Movie, Rating

DATA_DIR = os.environ.get("DATA_DIR", "../data/processed")
CHUNK_SIZE = 20_000


def main(force: bool = False) -> None:
    Base.metadata.create_all(engine)

    db = SessionLocal()
    try:
        existing = db.query(func.count(Movie.movie_idx)).scalar() or 0
        if existing and not force:
            ratings_count = db.query(func.count(Rating.id)).scalar() or 0
            print(f"Database already seeded ({existing} movies, {ratings_count} ratings) - skipping.")
            return

        movies_path = os.path.join(DATA_DIR, "final_movies.csv")
        ratings_path = os.path.join(DATA_DIR, "final_ratings.csv")
        for path in (movies_path, ratings_path):
            if not os.path.exists(path):
                sys.exit(
                    f"Missing {path}.\nRun 'python3 src/reconstruct_mappings.py' from the "
                    "project root first (requires data/raw/rating.csv and data/raw/movie.csv)."
                )

        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)

        db.execute(text("TRUNCATE TABLE ratings, movies RESTART IDENTITY CASCADE"))

        db.bulk_insert_mappings(
            Movie,
            [
                {
                    "movie_idx": int(row.MovieIdsNew),
                    "movie_id": int(row.movieId),
                    "title": row.title,
                    "genres": row.genres,
                }
                for row in movies.itertuples()
            ],
        )
        db.commit()
        print(f"Inserted {len(movies)} movies")

        records = ratings[["userId", "MovieIdsNew", "rating"]].rename(
            columns={"userId": "user_idx", "MovieIdsNew": "movie_idx"}
        )
        total = 0
        for start in range(0, len(records), CHUNK_SIZE):
            chunk = records.iloc[start : start + CHUNK_SIZE]
            db.bulk_insert_mappings(Rating, chunk.to_dict("records"))
            db.commit()
            total += len(chunk)
            print(f"  inserted {total}/{len(records)} ratings", end="\r")
        print(f"\nInserted {total} ratings")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-seed even if data exists")
    main(force=parser.parse_args().force)

"""
Reconstructs the userId/MovieIdsNew mappings that ncf_model.keras was trained on,
without retraining. Mirrors the exact deterministic logic of preprocess.py +
preprocess_shrink.py so the recovered mapping lines up with the model's saved
embedding weights.

Outputs:
  data/processed/final_ratings.csv  (userId, MovieIdsNew, rating, movieId)
  data/processed/final_movies.csv   (MovieIdsNew, movieId, title, genres)
"""
import pandas as pd
import numpy as np
import os

RAW = "data/raw"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)

# ---- mirror preprocess.py ----
chunks = pd.read_csv(os.path.join(RAW, "rating.csv"), chunksize=100000)
df = []
for chunk in chunks:
    df.append(chunk)
    if len(df) >= 5:
        break
df = pd.concat(df)
df.drop("timestamp", inplace=True, axis=1)

df["userId"] = df["userId"] - 1

unique_movie_ids = sorted(df["movieId"].unique())
movieIdMap = {val: idx for idx, val in enumerate(unique_movie_ids)}
df["movieIdsMapped"] = df["movieId"].map(movieIdMap)

# ---- mirror preprocess_shrink.py ----
N = len(df["userId"].unique())
M = len(df["movieIdsMapped"].unique())
n, m = 10000, 2000

user_ids, user_freq = np.unique(df["userId"], return_counts=True)
user_dict = {user_ids[i]: user_freq[i] for i in range(N)}
top_10000_users = sorted(user_dict.items(), key=lambda x: x[1], reverse=True)[:n]

movie_ids, movie_freq = np.unique(df["movieIdsMapped"], return_counts=True)
movie_dict = {movie_ids[i]: movie_freq[i] for i in range(M)}
top_2000_movies = sorted(movie_dict.items(), key=lambda x: x[1], reverse=True)[:m]

top_user_ids = set(dict(top_10000_users).keys())
top_movie_ids = set(dict(top_2000_movies).keys())

df_small = df[df["userId"].isin(top_user_ids) & df["movieIdsMapped"].isin(top_movie_ids)].copy()

unique_users = sorted(df_small["userId"].unique())
new_user_id_map = {org: new for new, org in enumerate(unique_users)}
df_small["userIdsNew"] = df_small["userId"].map(new_user_id_map)

unique_movies = sorted(df_small["movieIdsMapped"].unique())
new_movie_id_map = {org: new for new, org in enumerate(unique_movies)}
df_small["MovieIdsNew"] = df_small["movieIdsMapped"].map(new_movie_id_map)

df_small = df_small.drop(columns=["userId", "movieIdsMapped"])
df_small = df_small.rename(columns={"userIdsNew": "userId"})
# columns now: movieId (original), rating, userId (final), MovieIdsNew (final)

N_final = df_small["userId"].max() + 1
M_final = df_small["MovieIdsNew"].max() + 1
print(f"Reconstructed N (users) = {N_final}, M (movies) = {M_final}")
print("Expected from saved model: N=3374, M=2000")
assert N_final == 3374, f"user count mismatch: {N_final} != 3374"
assert M_final == 2000, f"movie count mismatch: {M_final} != 2000"
print("Mapping reconstruction verified against trained model's embedding shapes.")

df_small.to_csv(os.path.join(OUT, "final_ratings.csv"), index=False)

# ---- attach real titles ----
movies = pd.read_csv(os.path.join(RAW, "movie.csv"))
id_map = df_small[["MovieIdsNew", "movieId"]].drop_duplicates().sort_values("MovieIdsNew")
final_movies = id_map.merge(movies, on="movieId", how="left")
final_movies.to_csv(os.path.join(OUT, "final_movies.csv"), index=False)

print(f"Wrote {len(df_small)} ratings and {len(final_movies)} movies.")
print(final_movies.head(10)[["MovieIdsNew", "movieId", "title", "genres"]])

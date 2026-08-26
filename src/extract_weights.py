"""
Extracts trained weights from ncf_model.keras into a plain .npz file.

Why: this environment's Python has no compatible TensorFlow build, so instead
of loading the model via tf.keras, we read its architecture graph (config.json)
and weight arrays (model.weights.h5) directly out of the .keras zip archive and
reimplement the forward pass in NumPy (see backend/app/ncf_model.py). The layer
connectivity was verified against config.json's inbound_nodes graph:

  GMF branch:  embedding (user) x embedding_1 (movie) -> elementwise multiply
  MLP branch:  embedding_2 (user) | embedding_3 (movie) -> concat -> Dense(64,relu)
               -> Dense(32,relu)
  Fusion:      concat(gmf, mlp) -> Dense(1, linear)
"""
import zipfile
import json
import io
import h5py
import numpy as np
import pandas as pd

MODEL_PATH = "models/neural collaborative filtering/saved model/ncf_model.keras"
OUT_PATH = "models/neural collaborative filtering/saved model/ncf_weights.npz"

z = zipfile.ZipFile(MODEL_PATH)
weights_h5 = h5py.File(io.BytesIO(z.read("model.weights.h5")), "r")

def w(layer, idx):
    return np.array(weights_h5[f"layers/{layer}/vars/{idx}"])

emb_user_gmf = w("embedding", 0)      # (N, k)
emb_movie_gmf = w("embedding_1", 0)   # (M, k)
emb_user_mlp = w("embedding_2", 0)    # (N, k)
emb_movie_mlp = w("embedding_3", 0)   # (M, k)

dense_w, dense_b = w("dense", 0), w("dense", 1)       # (30,64), (64,)
dense1_w, dense1_b = w("dense_1", 0), w("dense_1", 1) # (64,32), (32,)
dense2_w, dense2_b = w("dense_2", 0), w("dense_2", 1) # (47,1), (1,)

ratings = pd.read_csv("data/processed/final_ratings.csv")
mu = ratings["rating"].mean()

np.savez(
    OUT_PATH,
    emb_user_gmf=emb_user_gmf,
    emb_movie_gmf=emb_movie_gmf,
    emb_user_mlp=emb_user_mlp,
    emb_movie_mlp=emb_movie_mlp,
    dense_w=dense_w, dense_b=dense_b,
    dense1_w=dense1_w, dense1_b=dense1_b,
    dense2_w=dense2_w, dense2_b=dense2_b,
    mu=np.array(mu),
    N=np.array(emb_user_gmf.shape[0]),
    M=np.array(emb_movie_gmf.shape[0]),
)
print(f"Saved weights to {OUT_PATH}")
print(f"N={emb_user_gmf.shape[0]} M={emb_movie_gmf.shape[0]} k={emb_user_gmf.shape[1]} mu={mu:.4f}")

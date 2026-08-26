"""
NumPy reimplementation of the trained NeuMF (Neural Collaborative Filtering)
forward pass. Loads weights extracted from ncf_model.keras by
src/extract_weights.py and reproduces:

  GMF branch:  user_emb * movie_emb (elementwise)
  MLP branch:  concat(user_emb, movie_emb) -> Dense(64, relu) -> Dense(32, relu)
  Fusion:      concat(gmf, mlp) -> Dense(1, linear) + mu

Verified against the model's own config.json connectivity graph and its
embedding/dense weight shapes.
"""
import os
from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "neural collaborative filtering"
    / "saved model"
    / "ncf_weights.npz"
)
WEIGHTS_PATH = Path(os.environ.get("NCF_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH))


class NCFModel:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        d = np.load(weights_path)
        self.emb_user_gmf = d["emb_user_gmf"]
        self.emb_movie_gmf = d["emb_movie_gmf"]
        self.emb_user_mlp = d["emb_user_mlp"]
        self.emb_movie_mlp = d["emb_movie_mlp"]
        self.dense_w, self.dense_b = d["dense_w"], d["dense_b"]
        self.dense1_w, self.dense1_b = d["dense1_w"], d["dense1_b"]
        self.dense2_w, self.dense2_b = d["dense2_w"], d["dense2_b"]
        self.mu = float(d["mu"])
        self.n_users = int(d["N"])
        self.n_movies = int(d["M"])

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Predicted rating for each (user_id, movie_id) pair, same length arrays."""
        u_gmf = self.emb_user_gmf[user_ids]
        m_gmf = self.emb_movie_gmf[movie_ids]
        gmf = u_gmf * m_gmf

        u_mlp = self.emb_user_mlp[user_ids]
        m_mlp = self.emb_movie_mlp[movie_ids]
        mlp = np.concatenate([u_mlp, m_mlp], axis=-1)
        mlp = self._relu(mlp @ self.dense_w + self.dense_b)
        mlp = self._relu(mlp @ self.dense1_w + self.dense1_b)

        fusion = np.concatenate([gmf, mlp], axis=-1)
        out = fusion @ self.dense2_w + self.dense2_b
        return out[..., 0] + self.mu

    def recommend_for_user(self, user_id: int, exclude_movie_ids: set[int], top_k: int = 10):
        """Top-k (movie_id, predicted_rating) pairs for a user, excluding already-rated movies."""
        candidates = np.array([m for m in range(self.n_movies) if m not in exclude_movie_ids])
        if len(candidates) == 0:
            return []
        scores = self.predict(np.full(len(candidates), user_id), candidates)
        order = np.argsort(-scores)[:top_k]
        return [(int(candidates[i]), float(scores[i])) for i in order]


_model: NCFModel | None = None


def get_model() -> NCFModel:
    global _model
    if _model is None:
        _model = NCFModel()
    return _model

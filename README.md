## Dataset

Raw data is not included due to size.

### Steps to generate processed dataset:

1. Download the MovieLens 2M dataset
2. Place files in `data/raw/`
3. Run preprocessing scripts from `src/`
4. Processed files will be saved to `data/processed/`

---

## Collaborative Filtering (Baseline Model)

This project implements **User-Based Collaborative Filtering** as a baseline recommendation system.

### Approach:

* Compute similarity between users (e.g. cosine similarity)
* Predict ratings based on similar users' preferences
* Evaluate using Mean Squared Error (MSE)

### Results:

* Train MSE: ~0.593
* Test MSE: ~0.666

---
## Dataset

Raw data is not included due to size.

### Steps to generate processed dataset:

1. Download the MovieLens 2M dataset
2. Place files in `data/raw/`
3. Run preprocessing scripts from `src/`
4. Processed files will be saved to `data/processed/`

---

## Collaborative Filtering (Baseline Model)

This project implements **User-Based Collaborative Filtering** as a baseline recommendation system.

### Approach:

* Compute similarity between users (e.g. cosine similarity)
* Predict ratings based on similar users' preferences
* Evaluate using Mean Squared Error (MSE)

### Results:

* Train MSE: ~0.593
* Test MSE: ~0.666

---

## Matrix Factorization

This project implements **Matrix Factorization with bias and regularization** using **Alternating Least Squares (ALS)**.

### Approach:

* Decompose user-item interaction matrix into latent user and item factors
* Incorporate:

  * Global mean (μ)
  * User bias (bᵤ)
  * Item bias (bᵢ)
* Optimize using **Alternating Least Squares (ALS)**

  * Alternately fix user factors and solve for item factors, and vice versa
* Evaluate using Mean Squared Error (MSE)

### Results:

* Train MSE: ~0.502
* Test MSE: ~0.540

### Improvement over Baseline:

* CF RMSE: ~0.82
* MF RMSE: ~0.73

This demonstrates a significant improvement in recommendation accuracy using latent factor models.

---

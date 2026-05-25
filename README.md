## Dataset

Raw data is not included due to size.

### Steps to generate processed dataset:

1. Download the **MovieLens 2M dataset**
2. Place files in `data/raw/`
3. Run preprocessing scripts from `src/`
4. Processed files will be saved to `data/processed/`

---

# Recommendation Models Implemented

This project explores multiple recommendation approaches, from traditional collaborative filtering to neural recommendation models.

---

## 1. Collaborative Filtering (Baseline Model)

This project implements **User-Based Collaborative Filtering** as a baseline recommendation system.

### Approach

* Compute similarity between users (e.g. cosine similarity)
* Predict ratings based on similar users' preferences
* Evaluate using Mean Squared Error (MSE)

### Results

* **Train MSE:** ~0.593
* **Test MSE:** ~0.666
* **RMSE:** ~0.82

---

## 2. Matrix Factorization

This project implements **Matrix Factorization with bias and regularization** using **Alternating Least Squares (ALS)**.

### Approach

* Decompose the user-item interaction matrix into latent user and item factors
* Incorporate:

  * Global mean (**μ**)
  * User bias (**bᵤ**)
  * Item bias (**bᵢ**)

* Optimize using **Alternating Least Squares (ALS)**

  * Alternately fix user factors and solve for item factors
  * Fix item factors and solve for user factors

* Evaluate using Mean Squared Error (MSE)

### Results

* **Train MSE:** ~0.502
* **Test MSE:** ~0.540
* **RMSE:** ~0.73

### Improvement over Baseline

* Collaborative Filtering RMSE: ~0.82
* Matrix Factorization RMSE: ~0.73

This demonstrates significant improvement in recommendation accuracy using latent factor models.

---

## 3. Neural Matrix Factorization (Keras)

Implements **Matrix Factorization using neural embeddings and gradient descent** in Keras.

### Approach

* Learn user and movie latent embeddings
* Predict ratings using:

  r̂(u, i) = Uᵤ · Wᵢ + bᵤ + bᵢ

* Train using Mean Squared Error (MSE)

### Model Architecture

* Inputs:

  * `userId`
  * `movieId`

* Layers:

  * User embedding
  * Movie embedding
  * Dot product interaction
  * User bias
  * Movie bias
  * Output prediction layer

### Results

* **Train MSE:** ~0.563
* **Test MSE:** ~0.568
* **RMSE:** ~0.75

---

## 4. Neural Collaborative Filtering (NeuMF)

Implements **Neural Collaborative Filtering (NCF)** using the **NeuMF architecture**, combining **Generalized Matrix Factorization (GMF)** and **Multi-Layer Perceptron (MLP)**.

### Approach

* Learn separate embeddings for:

  * GMF branch
  * MLP branch

* GMF branch captures **linear interactions**

  * Element-wise multiplication of user and movie embeddings

* MLP branch captures **non-linear interactions**

  * Concatenation of embeddings followed by a feedforward neural network

* Combine both branches for final rating prediction

* Train using:

  * Mean Squared Error (MSE)
  * Dropout regularization
  * L2 regularization
  * Early stopping

### Model Architecture

#### Inputs

* `userId`
* `movieId`

#### GMF branch

* User embedding
* Movie embedding
* Element-wise multiplication

#### MLP branch

* User embedding
* Movie embedding
* Concatenation
* Dense(64) → ReLU
* Dropout(0.2)
* Dense(32) → ReLU
* Dropout(0.2)

#### Fusion

* Concatenate GMF and MLP outputs
* Final Dense layer for rating prediction

### Results

* **Train MSE:** ~0.670
* **Test MSE:** ~0.685
* **RMSE:** ~0.83

### Observations

* Stable training with controlled overfitting
* Validation loss converges smoothly
* Combines strengths of linear and non-linear collaborative filtering
* Provides a deep learning-based recommendation baseline for hybrid recommendation systems

---

# Model Comparison

| Model | Test MSE | RMSE |
|---|---|---|
| User-Based Collaborative Filtering | 0.666 | 0.82 |
| Matrix Factorization (ALS) | 0.540 | 0.73 |
| Neural Matrix Factorization | 0.568 | 0.75 |
| Neural Collaborative Filtering (NeuMF) | 0.685 | 0.83 |

---




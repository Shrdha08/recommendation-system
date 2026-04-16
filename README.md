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

These results show:

* Good generalization (low gap between train and test error)
* A strong baseline for further improvement

---

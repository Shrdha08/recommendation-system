import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix, save_npz

#loading data
print("Loading data...")
df = pd.read_csv('data/processed/preprocessed_top_rating.csv')

N = df.userId.max() + 1  # number of users
M = df.movieId.max() + 1  # number of movies

#train - test split
df = shuffle(df, random_state=42)

cutoff = int(0.8 * len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# create train matrix
rows = df_train['userId'].values
cols = df_train['movieId'].values
data = df_train['rating'].values

A = csr_matrix((data, (rows, cols)), shape=(N, M))

# mask: where ratings exist
mask = (A > 0)

# save
save_npz("data/processed/Atrain.npz", A)

#create test matrix
rows_test = df_test['userId'].values
cols_test = df_test['movieId'].values
data_test = df_test['rating'].values

A_test = csr_matrix((data_test, (rows_test, cols_test)), shape=(N, M))

# mask: where ratings exist
mask_test = (A_test > 0)

# save
save_npz("data/processed/Atest.npz", A_test)
print('done')

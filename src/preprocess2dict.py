import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

df=pd.read_csv('data/processed/preprocessed_top_rating.csv',nrows=100000)

df['userId'] = df['userId'].astype(np.int32)
df['MovieIdsNew'] = df['MovieIdsNew'].astype(np.int32)
df['rating'] = df['rating'].astype(np.float32)

N=df['userId'].max()+1
M=df['MovieIdsNew'].max()+1

# df=shuffle(df)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
cutoff=int(0.8*len(df))
df_train=df.iloc[:cutoff]
df_test=df.iloc[cutoff:]

user2movie={}
movie2user={}
usermovie2rating={}

def update_user2movie_and_movie2user(row):
    i=int(row['userId'])
    j=int(row['MovieIdsNew'])

    if i not in user2movie:
        user2movie[i]=[j]
    else:
        user2movie[i].append(j)

    if j not in movie2user:
        movie2user[j]=[i]
    else:
        movie2user[j].append(i)

    usermovie2rating[(i,j)]=row['rating']

for _, row in df_train.iterrows():
    update_user2movie_and_movie2user(row)

usermovie2rating_test={}

def update_usermovie2rating_test(row):
    i=int(row['userId'])
    j=int(row['MovieIdsNew'])
    usermovie2rating_test[(i,j)]=row['rating']

for _, row in df_test.iterrows():
    update_usermovie2rating_test(row)

with open('data/processed/user2movie.pkl','wb') as f:
    pickle.dump(user2movie,f)

with open('data/processed/movie2user.pkl','wb') as f:
    pickle.dump(movie2user,f)

with open('data/processed/usermovie2rating.pkl','wb') as f:
    pickle.dump(usermovie2rating,f)

with open('data/processed/usermovie2rating_test.pkl','wb') as f:
    pickle.dump(usermovie2rating_test,f)
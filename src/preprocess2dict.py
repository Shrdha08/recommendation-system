import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

df=pd.read_csv('data/raw/preprocessed_top_rating.csv')

N=df['userId'].max()+1
M=df['movieId'].max()+1

df=shuffle(df)
cutoff=int(0.8*len(df))
df_train=df.iloc[:cutoff]
df_test=df.iloc[cutoff:]

user2movie={}
movie2user={}
usermovie2rating={}

def update_user2movie_and_movie2user(row):
    i=int(row['userId'])
    j=int(row['movieId'])

    if i not in user2movie:
        user2movie[i]=[j]
    else:
        user2movie[i].append(j)

    if j not in movie2user:
        movie2user[j]=[i]
    else:
        movie2user[j].append(i)

    usermovie2rating[(i,j)]=row['rating']

df_train.apply(update_user2movie_and_movie2user,axis=1)

usermovie2rating_test={}

def update_usermovie2rating_test(row):
    i=int(row['userId'])
    j=int(row['movieId'])
    usermovie2rating_test[(i,j)]=row['rating']

df_test.apply(update_usermovie2rating_test,axis=1)

with open('data/raw/user2movie.pkl','wb') as f:
    pickle.dump(user2movie,f)

with open('data/raw/movie2user.pkl','wb') as f:
    pickle.dump(movie2user,f)

with open('data/raw/usermovie2rating.pkl','wb') as f:
    pickle.dump(usermovie2rating,f)

with open('data/raw/usermovie2rating_test.pkl','wb') as f:
    pickle.dump(usermovie2rating_test,f)
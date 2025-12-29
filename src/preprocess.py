import pandas as pd
import numpy as np

df=pd.read_csv("data/raw/rating.csv")

#drop timestamp column - not useful
df.drop('timestamp',inplace=True,axis=1)

# start userIds from 0 (already sequential in nature)
df['userId']=df['userId']-1

# map movie ids to sequential movieIds
# df['movieId'].values #gives an array
unique_movie_ids=set(df['movieId'].values)

cnt = 0

mp={}

for id in unique_movie_ids:
    mp[id]=cnt
    cnt+=1

df['movieIdsMapped']=df['movieId'].map(mp)

df.drop('movieId',inplace=True,axis=1)

# save modified csv
df.to_csv('data/processed/preprocessed_ratings.csv',index=False)
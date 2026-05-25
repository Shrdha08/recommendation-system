import pandas as pd
import numpy as np
import pickle

#reduce number of users to top 10000 users and top 2000 movies
df=pd.read_csv('data/processed/preprocessed_ratings.csv')

# total user and movie ids
N=len(df['userId'].unique())
M=len(df['movieIdsMapped'].unique())

n=10000
m=2000

user_ids,user_freq=np.unique(df['userId'],return_counts=True)

user_dict={}
for i in range (0,N):
    user_dict[user_ids[i]]=user_freq[i]


sorted_users = sorted(
    user_dict.items(),
    key=lambda x: x[1],   # sort by frequency
    reverse=True
)
top_10000_users = sorted_users[:n]


movie_ids,movie_freq=np.unique(df['movieIdsMapped'],return_counts=True)
movie_dict={}

for i in range(0,M):
    movie_dict[movie_ids[i]]=movie_freq[i]

sorted_movies=sorted(
    movie_dict.items(),
    key=lambda x:x[1],
    reverse=True
)

top_2000_movies=sorted_movies[:m]

top_users_dict=dict(top_10000_users)
top_movies_dict=dict(top_2000_movies)

top_user_ids=set(top_users_dict.keys())
top_movie_ids=set(top_movies_dict.keys())

df_small=df[df['userId'].isin(top_user_ids) &df['movieIdsMapped'].isin(top_movie_ids)].copy()

#map user ids to new ids + same for movies
unique_users = sorted(df_small['userId'].unique())
new_user_id_map = {org: new for new, org in enumerate(unique_users)}
new_to_org_map = {value:key for key,value in new_user_id_map.items()}

df_small['userIdsNew']=df_small['userId'].map(new_user_id_map)

unique_movies = sorted(df_small['movieIdsMapped'].unique())
new_movie_id_map = {org: new for new, org in enumerate(unique_movies)}
new_to_org_movie={value:key for key,value in new_movie_id_map.items()}


df_small['MovieIdsNew']=df_small['movieIdsMapped'].map(new_movie_id_map)

df_small=df_small.drop(columns=['userId','movieIdsMapped'],axis=1)
df_small=df_small.rename(columns={
    'userIdsNew':'userId'
})

df_small.to_csv('data/processed/preprocessed_top_rating.csv',index=False)
with open('data/processed/user_map.pkl','wb') as f:
    pickle.dump(new_user_id_map,f)

with open('data/processed/movie_map.pkl','wb') as f:
    pickle.dump(new_movie_id_map,f)

with open ('models/neural collaborative filtering/saved model/moviemapping.pkl','wb') as f:
    pickle.dump(new_to_org_movie,f)
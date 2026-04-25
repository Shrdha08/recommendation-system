import pandas as pd
import numpy as np
import pickle
import time

print('start')
print('rating')

start_time = time.time()
chunks = pd.read_csv("data/raw/rating.csv", chunksize=100000)

df = []
for chunk in chunks:
    df.append(chunk)
    if len(df) >= 5:
        break

df = pd.concat(df)
print("done reading", time.time() - start_time)

print('tag')
tags=pd.read_csv('data/raw/tag.csv')
print('movie')
movies=pd.read_csv('data/raw/movie.csv')

print('0')
#drop timestamp column - not useful
df.drop('timestamp',inplace=True,axis=1)

movies['genres']=movies['genres'].str.replace('|',' ',regex=False)
# create new columns for tag based on userId and movieId
tags['tag'] = tags['tag'].fillna('').astype(str)
tag_agg = tags.groupby('movieId')['tag'].agg(' '.join).reset_index()
movies = movies.merge(tag_agg, on='movieId', how='left')
print('0')

movies['content'] = movies['genres'] + " " + movies['tag'].fillna('')
movies['content']=movies['content'].str.lower()

# start userIds from 0 (already sequential in nature)
df['userId']=df['userId']-1
print('0')

# map movie ids to sequential movieIds
# df['movieId'].values #gives an array
unique_movie_ids=sorted(df['movieId'].unique())

movieIdMap={val:idx for idx,val in enumerate(unique_movie_ids)}
new_to_original_mapping={mapped:org for org,mapped in movieIdMap.items()}

print('0')

df['movieIdsMapped']=df['movieId'].map(movieIdMap)

# save modified csv
df.to_csv('data/processed/preprocessed_ratings.csv',index=False)
movies.to_csv('data/processed/movies_with_content.csv', index=False)
print('0')

with open('data/processed/orgToNew.pkl','wb') as f:
    pickle.dump(movieIdMap,f)

with open('data/processed/newToOrg.pkl','wb') as f:
    pickle.dump(new_to_original_mapping,f)

print('done')
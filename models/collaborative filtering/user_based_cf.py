import pickle
import numpy as np
import pandas as pd
import os
if not os.path.exists('data/raw/user2movie.pkl') or \
not os.path.exists('data/raw/movie2user.pkl') or \
not os.path.exists('data/raw/usermovie2rating.pkl') or \
not os.path.exists('data/raw/usermovie2rating_test.pkl'):
    from src import preprocess2dict

# pickle file stores python object in binary form in a file- we take that file and then reconstruct python object (dictionary in this case) using pickle.load()
with open('data/raw/user2movie.pkl','rb') as f:
    user2movie=pickle.load(f)

with open('data/raw/movie2user.pkl','rb') as f:
    movie2user=pickle.load(f)

with open('data/raw/usermovie2rating.pkl','rb') as f:
    usermovie2rating=pickle.load(f)

with open('data/raw/usermovie2rating_test.pkl','rb') as f:
    usermovie2rating_test=pickle.load(f)

N = np.max(list(user2movie.keys())) + 1
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1

k=25 #to get k closest neighbours - k highest weights
limit=5 #users must have atleast 5 movies in common

#get each user's neighbours,each users avg rating,each user's
neighbors=[]
averages=[]
deviations=[]

for i in range(N):
    #find k closest users to user i
    movies_i=user2movie[i]
    movies_i_set=set(movies_i)

    #avg and deviations
    ratings_i={movie: usermovie2rating[(i,movie)] for movie in movies_i}
    avg_i=np.mean(list(ratings_i.values()))
    dev_i={movie:(rating-avg_i) for movie,rating in ratings_i.items()}
    dev_i_values=np.array(list(dev_i.values()))
    sigma_i=np.sqrt(dev_i_values.dot(dev_i_values))

    averages.append(avg_i)
    
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from sortedcontainers import SortedList

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

print('step 1 done')
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
    movies_i=user2movie[i]
    movies_i_set=set(movies_i)

    #calculate average & deviation
    ratings_i={movie:usermovie2rating[(i,movie)] for movie in movies_i}
    avg_i=np.mean(list(ratings_i.values()))
    dev_i={movie:rating-avg_i for movie,rating in ratings_i.items()}
    dev_i_values=np.array(list(dev_i.values()))
    sigma_i=np.sqrt(dev_i_values.dot(dev_i_values))#denominator in pearson correlation

    averages.append(avg_i)
    deviations.append(dev_i)

    sl=SortedList()
    for j in range(N):
        if j!=i:
            movies_j=user2movie[j]
            movies_j_set=set(movies_j)
            common_movies=(movies_i_set & movies_j_set)

            if len(common_movies)>limit:
                ratings_j={movie:usermovie2rating[(j,movie)] for movie in movies_j}
                avg_j=np.mean(list(ratings_j.values()))
                dev_j={movie:rating-avg_j for movie,rating in ratings_j.items()}
                dev_j_values=np.array(list(dev_j.values()))
                sigma_j=np.sqrt(dev_j_values.dot(dev_j_values))#denominator in pearson correlation

                w_ij=sum(dev_i[m]*dev_j[m] for m in common_movies)/(sigma_i*sigma_j)

                sl.add((-w_ij,j))
                if len(sl)>k:
                    del sl[-1]

    neighbors.append(sl)

print('step 2 done')
def predict(i,m):
    weight_sum=0
    deviation_sum=0
    
    for w,k in neighbors[i]:
        try:
            weight_sum+=-w
            deviation_sum+=(-w)*deviations[k][m]
        except KeyError:
            #neighbor has not rated same movie - dictionary has no value for user k and movie m
            pass

    if weight_sum==0:
        rating_pred=averages[i]
    else:
        rating_pred=averages[i]+deviation_sum/weight_sum

    rating_pred=min(5,rating_pred)
    rating_pred=max(0.5,rating_pred)
    return rating_pred

#perform predictions on training set
training_predict=[]
training_target=[]

for (i,m),rating in usermovie2rating:
    training_predict.append(predict(i,m))
    training_target.append(rating)

#repeat on test set
test_predict=[]
test_target=[]

for (i,m),rating in usermovie2rating_test:
    test_predict.append(predict(i,m))
    test_target.append(rating)

print('step 3 done')
#calculate accuracy
def mse(predict,target):
    return np.mean((np.array(target)-np.array(predict))**2)

train_mse=mse(training_predict,training_target)
test_mse=mse(test_predict,test_target)

print('train mse:',train_mse)
print('test mse:',test_mse)

print('all done')

    







    
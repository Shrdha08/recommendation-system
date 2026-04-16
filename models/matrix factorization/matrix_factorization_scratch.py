import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load in the data
import os
if not os.path.exists('data/processed/user2movie.pkl') or \
    not os.path.exists('data/processed/movie2user.pkl') or \
    not os.path.exists('data/processed/usermovie2rating.pkl') or \
    not os.path.exists('data/processed/usermovie2rating_test.pkl'):
    from src import preprocess2dict

with open('data/processed/user2movie.pkl','rb') as f:
    user2movies=pickle.load(f)

with open('data/processed/movie2user.pkl','rb') as f:
    movie2users=pickle.load(f)
    
with open('data/processed/usermovie2rating.pkl','rb') as f:
    usermovie2rating=pickle.load(f)

with open('data/processed/usermovie2rating_test.pkl','rb') as f:
    usermovie2rating_test=pickle.load(f)

# get total number of users = N
# get total number of movies = M

N=max(list(user2movies.keys()))+1
m1=max(list(movie2users.keys())) #movie2users has only movies from train set and not test set
m2=max([m for (u,m),r in usermovie2rating_test.items()])
M=max(m1,m2)+1

# prediction[i,j]=W[i].dot(U[j]) + b[i] + c[j]+mu
'''where
W[i] = latent vector for user i
U[j] = latent vector for movie j
b[i] → User bias
c[j] → Movie bias
mu → Global average (Average rating across all users and movies)

Loss function
minimize squared error + regularization:
L = summ((rating[i,j]-pred[i,j])^2 + λ(∥W[i]​∥^2+∥U[j​]∥^2+b[i]^2​+c[j]^2​))
'''

# initialising variables
k=10
W=np.random.randn(N,k)
U=np.random.randn(M,k)
b=np.zeros(N)
c=np.zeros(M)
mu=np.mean(list(usermovie2rating.values()))

#mean square error
def get_loss(d):
    # d: (user_id,movie_id)->rating
    sse=0
    N=float(len(d))

    for (i,j),rating in d.items():
        pred=W[i].dot(U[j])+b[i]+c[j]+mu
        sse += (pred-rating)*(pred-rating)
    
    return sse/N

epochs=25
reg=0.01
train_loss=[]
test_loss=[]

# model training - updating W[i],U[j],c[j],b[i]
for epoch in range(epochs):
    for i in range(N):
        matrix=np.eye(k)*reg
        vector=np.zeros(k)

        bi=0

        for j in user2movies[i]:
            rating=usermovie2rating[(i,j)]
            matrix+=np.outer(U[j],U[j])
            vector+=(rating-b[i]-c[j]-mu)*U[j]
            bi+=(rating-W[i].dot(U[j])-c[j]-mu)

        W[i]=np.linalg.solve(matrix,vector)
        b[i]=bi/(len(user2movies[i])+reg)

    for j in range(M):
        matrix=np.eye(k)*reg
        vector=np.zeros(k)

        cj=0
        try:
            for i in movie2users[j]:
                rating=usermovie2rating[(i,j)]
                matrix+=np.outer(W[i],W[i])
                vector+=(rating-b[i]-c[j]-mu)*W[i]
                cj+=(rating-W[i].dot(U[j])-b[i]-mu)

            U[j]=np.linalg.solve(matrix,vector)
            c[j]=cj/(len(movie2users[j])+reg)
        except KeyError:
            # movie has no rating
            pass

    # get training loss for epoch
    train_loss.append(get_loss(usermovie2rating))

    #get test loss for epoch
    test_loss.append(get_loss(usermovie2rating_test))

    print(f"epoch {epoch} complete")

print("training loss:",train_loss[-1])
print("test loss:",test_loss[-1])

print("train losses",train_loss)
print("test losses",test_loss)

print("train losses",train_loss)
plt.plot(train_loss,label="train_loss")
plt.plot(test_loss,label="test_loss")
plt.legend()
plt.show()






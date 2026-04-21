import pandas as pd
import numpy as np
from sklearn.utils import shuffle
# import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Embedding,Dot,Flatten,Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model

# model - r(i,j)=W[i].U[j]+b[i]+c[j]+mu
# average r(i,j) = r'(i,j)=r(i,j)-mu
# r'(i,j)=W[i].U[j]+b[i]+c[j]
# W,U learn as embeddings

# load data 
df=pd.read_csv('data/processed/preprocessed_top_rating.csv')
# rating,userId,movieId

# N = no of users(vocab size)
# M = no of movies
N=df['userId'].max()+1
M=df['movieId'].max()+1

#update label from rating to rating - global average
mu=df['rating'].mean()
df['avg_rating']=df['rating']-mu


# train-test split 80%,20%
df=shuffle(df)
cutoff=int(0.8*len(df))
df_train=df.iloc[:cutoff]
df_test=df.iloc[cutoff:]

# initalise variables
k=10 #dimension of embedding
epochs=25
reg=1e-5

'''
inputs - userId=u (N*1),movieId=m(M*1)
embedding (W,U) - U=embeddingU(N*K), W=embeddingM(M*k)
bias (b=movie_bias,c=user_bias) - b=embedding(N*1), c=embedding(M*1)
pred_value=W[i].U[j]+b[i]+c[j]
loss=MSE
'''

u=Input((1,))
m=Input((1,))
u_embedding=Embedding(N,k,embeddings_regularizer=l2(reg))(u)
m_embedding=Embedding(M,k,embeddings_regularizer=l2(reg))(m)
u_bias=Embedding(N,1,embeddings_regularizer=l2(reg))(u)
m_bias=Embedding(M,1,embeddings_regularizer=l2(reg))(m)

x = Dot(axes=2)([u_embedding, m_embedding])
x=Add()([x,u_bias,m_bias])
x=Flatten()(x)

model=Model(inputs=[u,m],outputs=x)
model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.001),
    metrics=['mse']
)

res=model.fit(
    x=[df_train['userId'].values,df_train['movieId'].values],
    y=df_train['avg_rating'].values,
    epochs=epochs,
    batch_size=128,
    validation_data=(
        [df_test['userId'].values,df_test['movieId'].values],
        df_test['avg_rating'].values
    )
)

plt.plot(res.history['loss'],label='train_loss')
plt.plot(res.history['val_loss'],label='test_loss')
plt.title('training loss VS testing loss')
plt.show()

plt.plot(res.history['mse'],label='train_mse')
plt.plot(res.history['val_mse'],label='test_mse')
plt.title('training mse VS testing mse')
plt.show()


import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Embedding,Dot,Flatten,Concatenate
from tensorflow.keras.layers import Multiply,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping

# W,U learn as embeddings
# then concatenated and passed onto a neural network

# load data 
df=pd.read_csv('data/processed/preprocessed_top_rating.csv')
# rating,userId,movieId

# N = no of users(vocab size)
# M = no of movies
N=df['userId'].max()+1
M=df['MovieIdsNew'].max()+1

print('length of df ' ,len(df))
#update label from rating to rating - global average
mu=df['rating'].mean()
df['avg_rating']=df['rating']-mu


# train-test split 80%,20%
df=shuffle(df)
cutoff=int(0.8*len(df))
df_train=df.iloc[:cutoff]
df_test=df.iloc[cutoff:]

# initalise variables
k=15 #dimension of embedding
epochs=25
reg=1e-4

# GMF
u=Input((1,))
m=Input((1,))
u_gmf=Embedding(N, k, embeddings_regularizer=l2(reg))(u)
m_gmf=Embedding(M, k, embeddings_regularizer=l2(reg))(m)

u_gmf=Flatten()(u_gmf)
m_gmf=Flatten()(m_gmf)

gmf=Multiply()([u_gmf, m_gmf])  # linear interaction

# MLP
u_mlp=Embedding(N, k, embeddings_regularizer=l2(reg))(u)
m_mlp=Embedding(M, k, embeddings_regularizer=l2(reg))(m)

u_mlp=Flatten()(u_mlp)
m_mlp=Flatten()(m_mlp)

mlp=Concatenate()([u_mlp, m_mlp])
# mlp=Dense(128, activation='relu')(mlp)
# mlp=Dropout(0.3)(mlp)
mlp=Dense(64, activation='relu')(mlp)
mlp=Dropout(0.2)(mlp)
mlp=Dense(32, activation='relu')(mlp)
mlp=Dropout(0.2)(mlp)

#COMBINE
x = Concatenate()([gmf, mlp])
x = Dense(1)(x)

model=Model(inputs=[u,m],outputs=x)
model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.001),
    metrics=['mse']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,             # wait 2 bad epochs
    restore_best_weights=True
)

res=model.fit(
    x=[df_train['userId'].values,df_train['MovieIdsNew'].values],
    y=df_train['avg_rating'].values,
    epochs=epochs,
    batch_size=128,
    validation_data=(
        [df_test['userId'].values,df_test['MovieIdsNew'].values],
        df_test['avg_rating'].values
    ),
    callbacks=[early_stop]
)

model.save('models/neural collaborative filtering/saved model/ncf_model.keras')
model_info={
    'mu':mu,
    'N':N,
    'M':M,
    'k':k
}

with open('models/neural collaborative filtering/saved model/ncf_metadata.pkl', 'wb') as f:
    pickle.dump(model_info, f)



plt.plot(res.history['loss'],label='train_loss')
plt.plot(res.history['val_loss'],label='test_loss')
plt.title('training loss VS testing loss')
plt.show()

plt.plot(res.history['mse'],label='train_mse')
plt.plot(res.history['val_mse'],label='test_mse')
plt.title('training mse VS testing mse')
plt.show()
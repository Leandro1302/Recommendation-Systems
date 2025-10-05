import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from __future__ import print_function, division
from builtins import range, input

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

df = pd.read_csv(r"..\large_files\movielens-20m-dataset\very_small_rating.csv", index_col = 0)

N = df.userId.max() + 1
M = df.movie_idx.max() + 1

df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# Variables
K = 10
mu = df_train.rating.mean()
epochs = 25
reg = 0.1
batch_size = 128

# Keras model
u = Input(shape=(1,)) # Batch size is implicit so this really means that your input is batch size by 1
m = Input(shape=(1,))
u_embd = Embedding(N, K, embeddings_regularizer=l2(reg))(u) # (Batch_size=N, 1, K) 
m_embd = Embedding(M, K, embeddings_regularizer=l2(reg))(m) # (Batch_size=M, 1, K)


u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u) # (Batch_size=N, 1, 1)
m_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(m) # (Batch_size=M, 1, 1)
x = Dot(axes = 2)([u_embd, m_embd]) # (Batch_size=N, 1, 1)
# axes = 2 specify the axes to sum over and we want to do over the K-size axes

x = Add()([x, u_bias, m_bias])
x = Flatten()(x) # (Batch_size=N, 1)

model = Model(inputs = [u,m], outputs = x)
model.compile(
    loss = "mse",
    optimizer=SGD(learning_rate= .01, momentum = .9),
    metrics=["mse"] # because we have also the reg
)

r = model.fit(
    x = [df_train.userId.values, df_train.movie_idx.values],
    y = df_train.rating.values - mu,
    epochs = epochs,
    batch_size=batch_size,
    validation_data=([df_test.userId.values, df_test.movie_idx.values],
                     df_test.rating.values - mu
    )
)

# Plots
plt.plot(r.history["loss"], label = "Train loss")
plt.plot(r.history["val_loss"], label = "Test loss")
plt.legend()
plt.show()

plt.plot(r.history["mse"], label = "Train MSE")
plt.plot(r.history["val_mse"], label = "Test MSE")
plt.legend()
plt.show()
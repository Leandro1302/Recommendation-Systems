import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from copy import deepcopy
from __future__ import print_function, division
from builtins import range, input
import os

if not os.path.exists("user2movies.json") or \
    not os.path.exists("movie2user.json") or \
    not os.path.exists("usermovie2rating.json") or \
    not os.path.exists("usermovie2rating_test.json"):
    import preprocess2dict

with open("user2movie.json", "rb") as f:
    user2movie = pickle.load(f)


with open("movie2user.json", "rb") as f:
    movie2user = pickle.load(f)


with open("usermovie2rating.json", "rb") as f:
    usermovie2rating = pickle.load(f)


with open("usermovie2rating_test.json", "rb") as f:
    usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys()))+1
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (_,m),_ in usermovie2rating_test.items()])
M = max(m1,m2)+1

print(f"N:{N}, M:{M}")

user2movierating = {}
movie2userrating = {}

for i,movies in user2movie.items():
    r = np.array([usermovie2rating[(i,j)] for j in movies])
    user2movierating[i] = (movies, r)

for j,users in movie2user.items():
    r = np.array([user2movierating(i,j)] for i in users)
    movie2userrating[j] = (users, r)


# we make the movie2userrating and not the user2movierating because the first is much faster
movie2userrating_test = {}

for (i,j), r in usermovie2rating_test.items():
    if j not in movie2userrating_test:
        movie2userrating_test[j] = ([i],[r])
    else:
        movie2userrating_test[j][0].append(i)
        movie2userrating_test[j][1].append(r)

for j, (users,r) in movie2userrating_test.items():
    movie2userrating_test[j][1] = np.array(r)


K = 10
W = np.random.randn(N,K)
U = np.random.randn(M,K)
b = np.zeros(N)
c = np.zeros(M)
mu = np.mean(list(user2movierating.values()))


def get_loss(m2u): # it will use t prediction where t is the number of users
    # d: movie_id -> (user_ids, ratings)
    N = 0
    sse = 0
    for j, (u_ids, r) in m2u.items():
        p = W[u_ids].dot(U[j]) + b[u_ids] + c[j] + mu # (TxK)(K)+(T)+(1)+(1) --> (Tx1)
        delta = p-r
        sse += delta.dot(delta)
        N += len(r)
    return sse/N

# Parameters
epochs = 25
reg = 0.1
train_losses = []
test_losses = []

for epoch in range(epochs):
    print(f"epoch n:{epoch}")
    epoch_start = datetime.now()
    # perform updates
    
    # update W and b
    t0 = datetime.now()
    for i in range(N):
        m_ids, r = user2movierating[i] # T
        matrix = U[m_ids].T.dot(U[m_ids]) + np.eye(K) * reg # (KxT)(TxK) + (KxK) -> (KxK)
        vector = (r - b[i] - c[m_ids] - mu).dot(U[m_ids]) # ((T) - (1) - (T) - (1))(TxK) --> (K)
        bi = (r - U[m_ids].dot(W[i]) - c[m_ids] - mu).sum() # ((T) - (TxK)(K) - (T) - (1)) -> (T)
        
        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi/(len(user2movie[i]+reg))
        
        if i%(N//10)==0:
            print(f"i:{i}, N:{N}")
    print(f"Updated W and b: {datetime.now()-t0}")
    
    
    # update U and c
    t0 = datetime.now()
    for j in range(M):
        try:
            u_ids, r = movie2userrating[j]
            matrix = W[u_ids].T.dot(W[u_ids]) + np.eye(K)*reg
            vector = (r - b[u_ids] - c[j] - mu).dot(W[u_ids])
            cj = (r - W[u_ids].dot(U[j]) - b[u_ids] - mu).sum()
            
            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj/len(movie2user[j] + reg)
            
            if j%(M//10)==0:
                print(f"j:{j}, M:{M}")
        except KeyError:
            pass
    print(f"updated U and c: {datetime.now()-t0}")
    print(f"epoch duration:{datetime.now()- epoch_start}")
    
    # Losses
    t0 = datetime.now()
    train_losses.append(get_loss(movie2userrating))
    test_losses.append(get_loss(movie2userrating_test))
    print(f"computation time:{datetime.now()-t0}")
    print(f"Train loss: {train_losses[-1]}")
    print(f"Test loss: {test_losses[-1]}")

print(f"Train losses: {train_losses}")
print(f"Test losses: {test_losses}")

# Plots
plt.plot(train_losses, label= "Train loss")
plt.plot(test_losses, label= "Test loss")
plt.legend()
plt.show()
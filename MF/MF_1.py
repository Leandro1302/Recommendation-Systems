import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList
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


# Initialize the parameters
K = 10
W = np.random.randn(N,K)
U = np.random.randn(M,K)
b = np.zeros(N)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))


def get_loss(d):

    N = float(len(d))
    sse =0

    for k,r in d.items():
        i,j = k
        p = W[i].dot(U[j])+b[i]+c[j]+mu
        sse += (p-r)*(p-r)

    return sse/N


# Training loop
epochs = 25
reg = 20.
train_losses = []
test_losses = []
for epoch in range(epochs):
    
    print(f"Epoch:{epoch}")
    epoch_start = datetime.now()

    # Update W and b
    t0 = datetime.now()
    for i in range(N):
        # For W
        matrix = np.eye(K) * reg
        vector = np.zeros(K)
        
        # For b
        bi = 0
        for j in user2movie[i]:
            r = usermovie2rating[(i,j)]
            matrix += np.outer(U[j], U[j])
            vector += (r-b[i]-c[j]-mu)*U[j]
            bi += (r-W[i].dot(U[j])-c[j]-mu)
        
        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi/len(user2movie[i]+reg)
        
        if i%(N//10)==0:
            print(f"i:{i}, N:{N}")
    print(f"updated W and b: {datetime.now()-t0}")


    # Update U and c
    t0 = datetime.now()
    for j in range(M):
        # For u
        matrix = np.eye(K) * reg
        vector = np.zeros(K)
        
        # For c
        cj = 0
        try:
            for i in movie2user[j]:
                r = usermovie2rating[(i,j)]
                matrix += np.outer(W[i], W[i])
                vector += (r-b[i]-c[j]-mu)*W[i]
                cj += (r-W[i].dot(U[j])-b[i]-mu)
            
            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / len(movie2user[j]+reg)
            
            if j%(M//10) == 0:
                print(f"j:{j}, M:{M}")
            
        except KeyError:
            pass
    
    print(f"updated U and c: {datetime.now()-t0}")
    print(f"epoch duration: {datetime.now()-epoch_start}")
                
    
    t0 = datetime.now()
    train_losses.append(get_loss(usermovie2rating))
    test_losses.append(get_loss(usermovie2rating_test))
    
    print(f"loss computation time:{datetime.now()-t0}")
    print(f"train loss: {train_losses[-1]}")
    print(f"test loss: {test_losses[-1]}")

print(f"train losses: {train_losses}")
print(f"test losses: {test_losses}")

plt.plot(train_losses, label = "Train losses")
plt.plot(test_losses, label = "Test losses")
plt.legend()
plt.show()
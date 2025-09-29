from collections import Counter
import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\Huawei\Documents\Corsi Udemy\Recommendation System\large_files\movielens-20m-dataset\edited_rating.csv")

N = df.userId.max()+1
M = df.movie_idx.max()+1

user_ids_count = Counter(df.userId)
movie_ids_count = Counter(df.movie_idx)

# Top 10000 user and Top 2000 films
n = 10000
m = 2000

user_ids = [u for u, _ in user_ids_count.most_common(n)]
movie_ids = [m for m, _ in movie_ids_count.most_common(m)]

df_small = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()

new_user_id_map = {}
i = 0
for old in user_ids:
    new_user_id_map[old] = i
    i+=1

print("i",i)

new_movie_id_map = {}
j = 0
for old in movie_ids:
    new_movie_id_map[old] = j
    j+=1

print("j",j)


df_small[:, "userId"] = df_small.apply(lambda row: new_user_id_map[row.userId], axis = 1)
df_small.loc[:, "movie_idx"] = df_small.apply(lambda row: new_movie_id_map[row.movie_idx], axis = 1)
print("max user id", df_small.userId.max())
print("max movie id", df_small.movie_idx.max())

df_small.to_csv(r"C:\Users\Huawei\Documents\Corsi Udemy\Recommendation System\large_files\movielens-20m-dataset\very_small_rating.csv")
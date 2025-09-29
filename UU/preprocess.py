import pandas as pd
df = pd.read_csv(r"C:\Users\Huawei\Documents\Corsi Udemy\Recommendation System\large_files\movielens-20m-dataset\rating.csv", sep = ",").iloc[:,:3]

df.userId = df.userId -1

# Adjust the indexes
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count +=1


df["movie_idx"] = df.apply(lambda row: movie2idx[row.movieId], axis = 1)

df.to_csv(r"C:\Users\Huawei\Documents\Corsi Udemy\Recommendation System\large_files\movielens-20m-dataset\edited_rating.csv")
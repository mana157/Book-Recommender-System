import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

print("Loading CSV files...")

books = pd.read_csv("Books.csv")
users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")

print("Creating popularity model...")

ratings_with_name = ratings.merge(books, on="ISBN")

num_rating_df = ratings_with_name.groupby("Book-Title").count()["Book-Rating"].reset_index()
num_rating_df.rename(columns={"Book-Rating": "num_ratings"}, inplace=True)

avg_rating_df = ratings_with_name.groupby("Book-Title")["Book-Rating"].mean().reset_index()
avg_rating_df.rename(columns={"Book-Rating": "avg_rating"}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
popular_df = popular_df[popular_df["num_ratings"] >= 250].sort_values(
    "avg_rating", ascending=False
).head(50)

popular_df = popular_df.merge(
    books, on="Book-Title"
).drop_duplicates("Book-Title")[
    ["Book-Title", "Book-Author", "Image-URL-M", "num_ratings", "avg_rating"]
]

print("Creating collaborative filtering model...")

x = ratings_with_name.groupby("User-ID").count()["Book-Rating"] > 200
padhe_likhe_users = x[x].index

filtered_rating = ratings_with_name[
    ratings_with_name["User-ID"].isin(padhe_likhe_users)
]

y = filtered_rating.groupby("Book-Title").count()["Book-Rating"] >= 50
famous_books = y[y].index

final_ratings = filtered_rating[
    filtered_rating["Book-Title"].isin(famous_books)
]

pt = final_ratings.pivot_table(
    index="Book-Title",
    columns="User-ID",
    values="Book-Rating"
)

pt.fillna(0, inplace=True)

similarity_scores = cosine_similarity(pt)

print("Saving PKL files...")

pickle.dump(popular_df, open("popular.pkl", "wb"))
pickle.dump(pt, open("pt.pkl", "wb"))
pickle.dump(books, open("books.pkl", "wb"))
pickle.dump(similarity_scores, open("similarity_scores.pkl", "wb"))

print("✅ Done! All PKL files generated successfully.")
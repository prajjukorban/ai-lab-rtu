import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


data = {
    'User': ['Alice', 'Alice', 'Bob', 'Bob', 'Carol', 'Carol', 'Dave', 'Eve'],
    'Movies': ['Avatar', 'Iron man', 'swadesh', 'Dangal', 'Pushpa', 'KGF', 'Harry Potter', 'Avengers'],
    'Rating': [5, 3, 4, 2, 5, 3, 4, 3]
}


df = pd.DataFrame(data)
user_post_matrix = df.pivot_table(index='User', columns='Movies', values='Rating').fillna(0)
scaler = StandardScaler()
normalized_matrix = scaler.fit_transform(user_post_matrix)
similarity = cosine_similarity(normalized_matrix)
similar_users = pd.DataFrame(similarity, index=user_post_matrix.index, columns=user_post_matrix.index)
print("User Similarity Matrix:")
print(similar_users)


def recommend_posts(target_user, top_n=2):
    similar_scores = similar_users[target_user].sort_values(ascending=False)
    similar_scores = similar_scores.drop(target_user)
    similar_user_indices = [user_post_matrix.index.get_loc(user) for user in similar_scores.index]
    weighted_ratings = np.dot(similar_scores.values, normalized_matrix[similar_user_indices])
    avg_ratings = weighted_ratings / similar_scores.sum()
    post_scores = pd.Series(avg_ratings, index=user_post_matrix.columns)
    rated_posts = user_post_matrix.loc[target_user]
    unrated_posts = rated_posts[rated_posts == 0].index
    recommendations = post_scores[unrated_posts].sort_values(ascending=False).head(top_n)
    return recommendations


recommended = recommend_posts("Eve", top_n=2)
print("\nRecommended posts for Eve:")
print(recommended)    
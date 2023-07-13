#!/usr/bin/env python
# coding: utf-8

# In[115]:


# Data processing
import pandas as pd
import numpy as np
import scipy.stats
# Visualization
import seaborn as sns
# Similarity
from sklearn.metrics.pairwise import cosine_similarity


# In[116]:


### https://medium.com/grabngoinfo/recommendation-system-user-based-collaborative-filtering-a2e76e3e15c4


# In[117]:


# Read in data
ratings=pd.read_csv('Downloads/ml-latest-small/ml-latest-small/ratings.csv')
# Take a look at the data
ratings.head()


# In[118]:


# Read in data
movies = pd.read_csv('Downloads/ml-latest-small/ml-latest-small/movies.csv')
# Take a look at the data
movies.head()


# In[119]:


# Merge ratings and movies datasets
df = pd.merge(ratings, movies, on='movieId', how='inner')
# Take a look at the data
df.head()


# In[120]:


# Aggregate by movie
agg_ratings = df.groupby('title').agg(mean_rating = ('rating', 'mean'),
                                                number_of_ratings = ('rating', 'count')).reset_index()
# Keep the movies with over 100 ratings
agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings']>100]
agg_ratings_GT100.info()


# In[121]:


# Check popular movies
agg_ratings_GT100.sort_values(by='number_of_ratings', ascending=False).head()


# In[122]:


# Merge data
df_GT100 = pd.merge(df, agg_ratings_GT100[['title']], on='title', how='inner')
df_GT100.info()


# In[123]:


df_GT100


# In[124]:


# Create user-item matrix
matrix = df_GT100.pivot_table(index='userId', columns='title', values='rating')
matrix.head()


# In[125]:


matrix.to_csv('Downloads/user_items_matrix.csv')


# In[126]:


# Normalize user-item matrix
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 'rows')
matrix_norm.head()


# In[127]:


# User similarity matrix using Pearson correlation
user_similarity = matrix_norm.T.corr()
user_similarity.head()


# In[130]:


user_similarity= user_similarity.fillna(0)
user_similarity


# In[131]:


# User similarity matrix using cosine similarity
user_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))
user_similarity_cosine


# # la matrice de similarite 

# In[132]:


user_similarity_cosine = pd.DataFrame(user_similarity_cosine, columns=matrix_norm.index, index=matrix_norm.index)
user_similarity_cosine


# # la matrice  ordonnée

# In[133]:


# Create a copy of the similarity matrix
sorted_similarity_matrix = np.copy(user_similarity_cosine)

# Sort the similarity matrix in descending order
sorted_similarity_matrix.sort(axis=0, kind='quicksort', order=None)
sorted_similarity_matrix = sorted_similarity_matrix[::-1]

# Convert the sorted similarity matrix to a DataFrame
df_sorted_similarity = pd.DataFrame(sorted_similarity_matrix, columns=matrix_norm.index, index=matrix_norm.index)

# Print the sorted similarity matrix as a DataFrame
df_sorted_similarity


# In[134]:


df_sorted_similarity.to_csv('Downloads/df_sorted_similarity3.csv')


# In[135]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define a range of cluster numbers to try
max_clusters = 20
sse = []

for num_clusters in range(1, max_clusters+1):
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(user_similarity_cosine)
    
    # Calculate the SSE
    sse.append(kmeans.inertia_)

# Plot the SSE values against the number of clusters
plt.plot(range(1, max_clusters+1), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.title('Elbow Method')
plt.show()


# In[136]:


from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

# Perform K-means clustering with a chosen number of clusters
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(user_similarity_cosine)

# Calculate the Silhouette scores
silhouette_avg = silhouette_score(user_similarity_cosine, clusters)
sample_silhouette_values = silhouette_samples(user_similarity_cosine, clusters)

# Print the Silhouette score and sample scores for each data point
print("Silhouette score:", silhouette_avg)
print("Sample Silhouette scores:")
for i, score in enumerate(sample_silhouette_values):
    print("Data point", i, ":", score)


# In[137]:


from sklearn.cluster import KMeans

# Define the number of clusters
k = 5

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(user_similarity_cosine)

# Get the cluster labels for all data points
predicted_labels = kmeans.labels_

# Print the cluster assignments and predicted labels
print("Cluster Assignments and Predicted Labels:")
for i, cluster in enumerate(clusters):
    print("Data point", i, " Cluster:", cluster, " Predicted Label:", predicted_labels[i])


# In[138]:


matrix['Cluster'] = clusters


# In[139]:


matrix


# In[140]:


matrix.fillna(0)


# In[112]:


# Get the cluster labels for all data points
predicted_labels = kmeans.labels_
predicted_labels


# In[142]:


# Get the cluster labels for all data points
predicted_labels = kmeans.labels_

# Plot the clusters
plt.scatter(user_similarity_cosine[:, 0], user_similarity_cosine[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.show()


# In[143]:


# Pick a user ID
picked_userid = 1
# Remove picked user ID from the candidate list
user_similarity.drop(index=picked_userid, inplace=True)
# Take a look at the data
user_similarity.head()


# In[81]:


# Number of similar users
n = 10
# User similarity threashold
user_similarity_threshold = 0.3
# Get top n similar users
similar_users = user_similarity[user_similarity[picked_userid]>user_similarity_threshold][picked_userid].sort_values(ascending=False)[:n]
# Print out top n similar users
print(f'The similar users for user {picked_userid} are', similar_users)


# In[82]:


# Movies that the target user has watched
picked_userid_watched = matrix_norm[matrix_norm.index == picked_userid].dropna(axis=1, how='all')
picked_userid_watched


# In[83]:


# Movies that similar users watched. Remove movies that none of the similar users have watched
similar_user_movies = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')
similar_user_movies


# In[84]:


# Remove the watched movie from the movie list
similar_user_movies.drop(picked_userid_watched.columns,axis=1, inplace=True, errors='ignore')
# Take a look at the data
similar_user_movies


# In[85]:


# A dictionary to store item scores
item_score = {}
# Loop through items
for i in similar_user_movies.columns:
  # Get the ratings for movie i
  movie_rating = similar_user_movies[i]
  # Create a variable to store the score
  total = 0
  # Create a variable to store the number of scores
  count = 0
  # Loop through similar users
  for u in similar_users.index:
    # If the movie has rating
    if pd.isna(movie_rating[u]) == False:
      # Score is the sum of user similarity score multiply by the movie rating
      score = similar_users[u] * movie_rating[u]
      # Add the score to the total score for the movie so far
      total += score
      # Add 1 to the count
      count +=1
  # Get the average score for the item
  item_score[i] = total / count
# Convert dictionary to pandas dataframe
item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])
    
# Sort the movies by score
ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)
# Select top m movies
m = 10
ranked_item_score.head(m)


# In[ ]:





# In[ ]:





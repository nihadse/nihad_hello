#!/usr/bin/env python
# coding: utf-8

# In[178]:


pip install kneed


# In[179]:


# Importing the required libraries for visualization 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# Visualization Prefrences.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[180]:


# to Pass the warning output in the results 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[181]:


# read by default 1st sheet of an excel file
df = pd.read_excel('Telco_customer_churn_services1.xlsx')
df


# In[182]:


df.columns


# In[183]:


df['Customer ID'].duplicated().sum()


# In[184]:


#### df['fixed_acidity'] = df['fixed_acidity'].apply(lambda x: [float(i) for i in str(x).split('|')])


# In[185]:


pd.set_option('display.max_columns', None)
df.head(15)


# In[186]:


df = df.drop(['Quarter', 'Paperless Billing','Payment Method','Referred a Friend'],axis = 1)


# In[187]:


df.columns


# In[188]:


df.info()


# In[189]:


from sklearn import preprocessing
features  = ['Customer ID','Offer', 'Phone Service','Multiple Lines',
       'Internet Service', 'Internet Type', 'Avg Monthly GB Download',
       'Online Security', 'Online Backup', 'Device Protection Plan',
       'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
       'Streaming Music', 'Unlimited Data', 'Contract']
le = preprocessing.LabelEncoder()

for i in features:
    df[i] = le.fit_transform(df[i])


# In[190]:


# Features normalizations 


# In[191]:


scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)


# # Kmeans

# #### elbow method

# In[192]:


kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse = []
for k in range(1, 25):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)


# In[193]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 25), sse)
plt.xticks(range(1, 25))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[194]:


kl = KneeLocator(range(1,25), sse, curve="convex", direction="decreasing")

kl.elbow


# #### Kmeans silhouette method

# In[195]:


# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 25):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[196]:


plt.style.use("fivethirtyeight")
plt.plot(range(2, 25), silhouette_coefficients)
plt.xticks(range(2, 25))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()


# In[197]:


kmeans = KMeans(n_clusters=6, random_state=0, n_init="auto").fit(df)
kmeans.labels_
kmeans.cluster_centers_


# In[198]:


df['clusters'] = kmeans.labels_


# In[199]:


df['clusters'].unique()


# # KNN

# In[200]:


from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[201]:


# Calculate the similarity matrix using KNN
k = 5  # Number of neighbors for KNN
knn = NearestNeighbors(n_neighbors=k)
knn.fit(df[features])
distances, indices = knn.kneighbors(df[features])


# In[202]:


# Find the optimal K using silhouette score
k_values = range(2, 10)  # Range of K values to evaluate
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(distances)
    cluster_labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(distances, cluster_labels))

# Find the best K value with the highest silhouette score
best_k = k_values[np.argmax(silhouette_scores)]

# Print the best K value and its corresponding silhouette score
print("Best K value:", best_k)
print("Silhouette score:", max(silhouette_scores))


# # The similarity matrix

# In[203]:


# Apply K-means clustering on the similarity matrix
K_clusters = 2  # Number of clusters for K-means
kmeans = KMeans(n_clusters=K_clusters)
kmeans.fit(distances)

# Get the cluster labels for each client
cluster_labels = kmeans.labels_

# Add cluster labels to the original dataframe
df['Cluster'] = cluster_labels

# Display the dataframe with cluster assignments
print(df)


# In[204]:


df['Cluster']


# In[205]:


###############
###########


# In[206]:


df


# #1 -  Normalize the data

# In[207]:


#1 -  Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df)


# In[208]:


# Define the distance metrics to use
distance_metrics = ['euclidean', 'manhattan', 'cosine', 'l1', 'l2', 'chebyshev', 'minkowski']

# Calculate the similarity matrix using KNN
k = 2  # Number of neighbors for KNN

# Iterate over each distance metric
for metric in distance_metrics:
    # Calculate the similarity matrix using the current distance metric
    if metric == 'cosine':
        # For cosine similarity, use pairwise_distances function
        distances = 1 - pairwise_distances(normalized_data, metric=metric)
    else:
        # For other metrics, use NearestNeighbors
        knn = NearestNeighbors(n_neighbors=k, metric=metric)
        knn.fit(normalized_data)
        distances, indices = knn.kneighbors(normalized_data)
        
    similarity_matrices[metric] = distances
    # Find the optimal K using silhouette score
    k_values = range(2, 10)  # Range of K values to evaluate
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(distances)
        cluster_labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(distances, cluster_labels))

    # Find the best K value with the highest silhouette score
    best_k = k_values[np.argmax(silhouette_scores)]
    

    # Print the results for the current distance metric
    print(f"Distance metric: {metric}")
    print("Best K value:", best_k)
    print("Silhouette score:", max(silhouette_scores))
    print("\n")
   # Iterate over each K value
    for k in range(2, 10):
        # Apply K-means clustering on the similarity matrix
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(distances)

        # Calculate evaluation metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(distances, cluster_labels)

        # Print the results for the current K value
        print(f"K = {k}, Inertia score: {inertia}, Silhouette score: {silhouette}")


# In[216]:


# Get the best performing result based on the highest Silhouette score (Euclidean)
best_metric = 'euclidean'
best_k = 2
best_distances = similarity_matrices['euclidean']

# Apply K-means clustering on the similarity matrix using the best metric and K value
kmeans = KMeans(n_clusters=best_k)
cluster_labels = kmeans.fit_predict(best_distances)


# In[251]:


best_distances.shape


# In[247]:


# Plot all the K-Means clusters
plt.scatter(best_distances[:, 0], best_distances[:, 1], c=cluster_labels)
plt.title(f"K-Means Clustering - {best_metric} (K = {best_k})")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()


# In[226]:


df['cluster_labels'] = cluster_labels


# In[249]:


# Print the similarity matrix for each distance metric
for metric, similarity_matrix in similarity_matrices.items():
    print(f"Similarity matrix using {metric} distance:")
    print(similarity_matrix)
    print("\n")


# In[237]:


df['cluster_labels'].unique()


# In[253]:


df.columns


# In[260]:


results = df[['Customer ID','cluster_labels']]
results


# In[262]:


import pandas as pd

# Load the transaction data into a DataFrame
data = {
    'Client ID': [1001, 1002, 1003, 1004],
    'Transaction Amount': [[50.35, 300.0], [100.2], [75.1], [500.75]],
    'Merchant Category': [['Restaurant', 'Electronics Store'], ['Clothing Store'], ['Grocery Store'], ['Travel Agency']],
    'Location': [['New York', 'Chicago'], ['Los Angeles'], ['Miami'], ['San Francisco']],
    'Transaction Date': [['2023-07-12', '2023-07-11'], ['2023-07-10'], ['2023-07-09'], ['2023-07-08']]
}
df = pd.DataFrame(data)

df


# In[263]:


df.columns


# In[268]:


df['Transaction Date'] = df['Transaction Date'].apply(lambda dates: [pd.to_datetime(date) for date in dates])
as


# In[273]:


df['Transaction Date'] = df['Transaction Date'].apply(lambda dates: [pd.to_datetime(date) for date in dates])
df['Transaction Date']


# In[277]:


# Convert the 'Transaction Date' column to datetime
df['Transaction Date'] = df['Transaction Date'].apply(lambda dates: [pd.to_datetime(date) for date in dates])

# Extract date features
df['DayOfWeek'] = df['Transaction Date'].apply(lambda dates: [date.day_name() for date in dates])
df['Month'] = df['Transaction Date'].apply(lambda dates: [date.month for date in dates])
df['Year'] = df['Transaction Date'].apply(lambda dates: [date.year for date in dates])

# Print the updated DataFrame
df


# In[284]:


# Extract the first date from the list in 'Transaction Date' column for grouping
df['Grouping Date'] = df['Transaction Date'].apply(lambda dates: dates[0])

# Calculate daily total transaction amount
daily_total = df.groupby('Grouping Date')['Transaction Amount'].sum()
daily_total


# In[285]:


df


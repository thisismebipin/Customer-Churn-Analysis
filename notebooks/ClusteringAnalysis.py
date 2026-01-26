#Load csv
import pandas as pd

df= pd.read_csv("Dataset_ATS_v2.csv")
df.head()
df.info()
df.isnull().sum()

# binary column by Label encoding
from sklearn.preprocessing import LabelEncoder

binary_column= ['gender','Dependents','PhoneService','MultipleLines','Churn']
le= LabelEncoder()
for col in binary_column:
    df[col]= le.fit_transform(df[col])

#for multi-class columns (One-hot encoding)

df= pd.get_dummies(df, columns=['InternetService','Contract'], drop_first=True)
df.info()

#df.to_csv("Processed_data.csv", index=False)


#Train/Test Split
from sklearn.model_selection import train_test_split

X=df.drop('Churn', axis=1) # Features
y= df['Churn'] # Target variables
#Split train and  test (80% Train and 20 % Test)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

# Combine features and target for training set
train_df = pd.concat([X_train, y_train], axis=1)

# Combine features and target for test set
test_df = pd.concat([X_test, y_test], axis=1)
'''
#Save train and test file
train_df.to_csv("train_set.csv", index=False)
test_df.to_csv("test_set.csv", index=False)
'''

# Feature Scaling using StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Findidng Optimal number of clusters(Elbow Method)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()
'''The Elbow appears around k=5, which suggests that 5 slusters is a reasonable and well balanced choice,
it captures most of the data structure without unnecessary complexity'''

# Check the validity of k usingf silhouette score

from sklearn.metrics import silhouette_score
silhouette_score(X_scaled, KMeans(n_clusters=5, random_state=42).fit_predict(X_scaled))
#Comparing wit k =3 and k= 4
for k in [3,4,5]:
    score = silhouette_score(
        X_scaled,
        KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)
    )
    print(f"k={k}, silhouette score={score:.3f}")
# Score is good for k=5 which is 0.168

#Apply K-Means with chosen k = 5
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['Cluster'] = clusters
#Size of each cluster
df['Cluster'].value_counts()
#Mean Profile for each cluster/ analysing cluster profile
cluster_profile = df.groupby('Cluster').mean()
print(cluster_profile)

print(df.groupby('Cluster')['Churn'].mean())

#Cluster Visualization
import matplotlib.pyplot as plt

churn_by_cluster = df.groupby('Cluster')['Churn'].mean()

churn_by_cluster.plot(kind='bar')
plt.title("Churn Rate by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Churn Rate")
plt.xticks(rotation=0)
plt.show()

#Cluster size distribution
cluster_counts = df['Cluster'].value_counts().sort_index()

cluster_counts.plot(kind='bar')
plt.title("Number of Customers per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Customer Count")
plt.xticks(rotation=0)
plt.show()

#Dimensionality reduction using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
scatter= plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df['Cluster'],
    cmap='viridis',
    alpha=0.7
)

# create legend to make visual clear

legend = plt.legend(
    *scatter.legend_elements(),
    title="Cluster",
    loc='best'
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Cluster Segmentation using PCA")
plt.show()
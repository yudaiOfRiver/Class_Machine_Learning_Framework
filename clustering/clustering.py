
#%%
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5,5),dpi=120)

df = pd.read_csv("sampleData1.csv")


print(df.isnull().any())#欠損値があるか
print("-----------------------")
print(df.isnull().sum())#欠損値の数


plt.scatter(df["X"],df["Y"])
plt.show()

#%%
#階層型クラスタリング
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def plot_dendrogram(model, **kwargs):

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)

    return linkage_matrix


model = AgglomerativeClustering(affinity='euclidean', linkage='ward',distance_threshold=0, n_clusters=None)

model = model.fit(df)
fig = plt.figure(figsize=(8,8),dpi=150)
linkage_matrix = plot_dendrogram(model, truncate_mode='level', p=4)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# %%
#シルエット係数によるクラスタ数の選択
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

clusterNumList = []
scoreList = []

for i in range(2,10):
    clusters = fcluster(linkage_matrix, t=i, criterion='maxclust')
    score = silhouette_score(df, clusters, metric='euclidean')
    print(i,score)
    clusterNumList.append(i)
    scoreList.append(score)

fig = plt.figure(figsize=(8,8))
plt.plot(clusterNumList,scoreList,marker="o")

# %%
#シルエット係数で最も良かったクラスタ数3で色分けして可視化

clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')

fig = plt.figure(figsize=(8,8))
plt.scatter(df["X"],df["Y"],c=clusters)

# %%
#非階層型クラスタリング:KMeans
from sklearn.cluster import KMeans

df = pd.read_csv("sampleData1.csv")

clusterNum = 3
model = KMeans(n_clusters=clusterNum, init='k-means++')
res = model.fit_predict(df)

df["cluster"] = res

fig = plt.figure(figsize=(5,5),dpi=120)
for i in range(clusterNum):
    plt.scatter(df[df["cluster"]==i]["X"],df[df["cluster"]==i]["Y"],label="c"+str(i))
plt.legend()

# %%
#非階層型クラスタリング:KMeans
#シルエット係数によるクラスタ数の選択
from sklearn.metrics import silhouette_score

clusterNumList = []
scoreList = []

for i in range(2,11):
    model = KMeans(n_clusters=i, init='k-means++')
    res = model.fit_predict(df)
    score = silhouette_score(df, res, metric='euclidean')
    print(i,score)
    clusterNumList.append(i)
    scoreList.append(score)

fig = plt.figure(figsize=(8,8))
plt.plot(clusterNumList,scoreList,marker="o")
# %%
#非階層型クラスタリング:KMeans
#エルボー法によるクラスタ数の選択


clusterNumList = []
scoreList = []

for i in range(2,11):
    model = KMeans(n_clusters=i, init='k-means++')
    res = model.fit_predict(df)
    clusterNumList.append(i)
    scoreList.append(model.inertia_)   # inertia_ に情報が入っている？

fig = plt.figure(figsize=(8,8))
plt.plot(clusterNumList,scoreList,marker="o")

# %%

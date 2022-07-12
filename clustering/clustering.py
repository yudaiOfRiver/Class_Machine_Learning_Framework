
#%%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sampleData1.csv")

print(df.isnull().any())
print("-----------------------")
print(df.isnull().sum())

fig1, ax1 = plt.subplots(figsize=(5,5),dpi=120)
ax1.scatter(df["X"],df["Y"])
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
fig1.savefig("fig1.png")

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
fig2, ax2 = plt.subplots(figsize=(8,8),dpi=150)
linkage_matrix = plot_dendrogram(model, truncate_mode='level', p=4)
ax2.set_xlabel("Number of points in node (or index of point if no parenthesis).")
fig2.show()
fig2.savefig("fig2.png")

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

fig3, ax3 = plt.subplots(figsize=(8,8))
ax3.plot(clusterNumList,scoreList,marker="o")
ax3.set_xlabel("The number of clusters")
ax3.set_ylabel("Silhouestte score")
fig3.savefig("fig3.png")


# %%
#シルエット係数で最も良かったクラスタ数3で色分けして可視化

clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')

fig4, ax4 = plt.subplots(figsize=(8,8))
ax4.scatter(df["X"],df["Y"],c=clusters)
ax4.set_xlabel("X")
ax4.set_ylabel("Y")
fig4.savefig("fig4.png")

# %%
#非階層型クラスタリング:KMeans
from sklearn.cluster import KMeans

df = pd.read_csv("sampleData1.csv")

clusterNum = 3
model = KMeans(n_clusters=clusterNum, init='k-means++')
res = model.fit_predict(df)

df["cluster"] = res

fig5, ax5 = plt.subplots(figsize=(5,5),dpi=120)
for i in range(clusterNum):
    ax5.scatter(df[df["cluster"]==i]["X"],df[df["cluster"]==i]["Y"],label="c"+str(i))

ax5.set_xlabel("X")
ax5.set_ylabel("Y")
fig5.legend()
fig5.savefig("fig5.png")
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

fig6, ax6 = plt.subplots(figsize=(8,8))
ax6.plot(clusterNumList,scoreList,marker="o")
ax6.set_xlabel("The number of clusters")
ax6.set_ylabel("Silhouestte score")
fig6.savefig("fig6.png")
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

fig7, ax7 = plt.subplots(figsize=(8,8))
ax7.plot(clusterNumList,scoreList,marker="o")
ax7.set_xlabel("The number of clusters")
ax7.set_ylabel("Elbow score")
fig7.savefig("fig7.png")


# %%

#%%
from matplotlib import projections
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("PCA-sampleData1.csv")
fig1 = plt.figure(figsize=(5,5),dpi=120)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

ax1.plot(df.X,df.Y,df.Z,marker="o",linestyle='None')
fig1.savefig("fig1.png")


# %%
#主成分分析
#説明率

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sc = StandardScaler()
df1 = pd.DataFrame(sc.fit_transform(df))  # 正規化?
df1.columns = df.columns

pca = PCA(n_components=3)
pca.fit(df1)
print(pca.components_)
print(pca.explained_variance_ratio_)

labelList = ["PC"+str(i+1) for i in range(len(pca.explained_variance_ratio_))]  # 3つの説明変数ごとに情報量を格納
fig2, ax2 = plt.subplots()
ax2.bar(labelList,pca.explained_variance_ratio_)
fig2.savefig("fig2.png")

# %%
#主成分空間へのマッピング

import numpy as np
import random

feature = pca.transform(df1)

fig3, ax3 = plt.subplots(figsize = (6,6),dpi=120)
ax3.scatter(feature[:,0],feature[:,1])
pc0 = pca.components_[0]
pc1 = pca.components_[1]

for i in range(pc0.shape[0]):
    ax3.arrow(0, 0, pc0[i]*0.5, pc1[i]*0.5, color='r', width=0.0005, head_width=0.1, alpha=0.9)
    ax3.text(pc0[i]*0.8, pc1[i]*0.8, df1.columns.values[i], color='r')
ax3.set_ylim(-3,3)
fig3.savefig("fig3.png")

# %%

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

boston = load_boston()
df = pd.DataFrame(boston.data,columns=boston.feature_names )
df["MEDV"] = boston.target

# %%
#単回帰モデル
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

x = df["RM"]
y = df["MEDV"]
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

# 学習
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model = LinearRegression()
model.fit(x_train, y_train)
a = model.coef_[0,0]
b = model.intercept_[0]
print("Gradient a = %f" % a)
print("Intercept b = %f" % b)
fig1, ax1 = plt.subplots(figsize=(5,5),dpi=100)
ax1.set_xlabel('RM')
ax1.set_ylabel('MEDV')
ax1.scatter(x_train,y_train,label="Train")
ax1.scatter(x_test,y_test,label="Test")
ax1.plot([4,9],[4*a+b,9*a+b],color = "red")
ax1.legend()
fig1.savefig("fig1.png")

#学習データに対するMAE (Mean Absolute Error)
y_pred = model.predict(x_train)
mae = mean_absolute_error(y_train, y_pred)
print("MAE for train data=",mae)

#評価データに対するMAE
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("MAE for test data=",mae)

# %%
#重回帰モデル
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

x = df.drop('MEDV', axis = 1)
y = df["MEDV"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

sc = StandardScaler()
x_train_std = sc.fit_transform(x_train) # 標準化

model2 = LinearRegression()
model2.fit(x_train_std, y_train)

#評価データに対するMAE
x_test_std = sc.transform(x_test) #学習データを標準化したときのパラメータを使って評価データの標準化
y_pred = model2.predict(x_test_std)
mae = mean_absolute_error(y_test, y_pred)
print("MAE for test data=",mae)


#実際の価格と予測した価格をプロット
fig2, ax2 = plt.subplots(figsize=(5,5),dpi=100)
ax2.set_xlabel("actual value")
ax2.set_ylabel("expected value")
ax2.scatter(y_test,y_pred)
ax2.plot([0,50],[0,50],color="red",ls="--") #一致した場合にプロットが乗る直線
fig2.savefig("fig2.png")

# %%

#多項式回帰
from sklearn.preprocessing import PolynomialFeatures

x = df["RM"]
y = df["MEDV"]
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

pf = PolynomialFeatures(degree=2, include_bias = False)
x_train_poly = pf.fit_transform(x_train)

model = LinearRegression()
model.fit(x_train_poly, y_train)
print(model.coef_)

#訓練した回帰線を描くための処理===
line = np.linspace(3,9,1000,endpoint=False).reshape(-1,1)
line_niji = pf.transform(line)

fig3, ax3 = plt.subplots(figsize=(5,5),dpi=100)
ax3.scatter(x_train,y_train,label="Train")
ax3.plot(line, model.predict(line_niji), color='red' )
ax3.set_xlabel("RM")
ax3.set_ylabel("MEDV")
fig3.savefig("fig3.png")


#学習データに対するMAE
y_pred = model.predict(x_train_poly)
mae = mean_absolute_error(y_train, y_pred)
print("MAE for train data=",mae)

#評価データに対するMAE
x_test_poly = pf.transform(x_test)
y_pred = model.predict(x_test_poly)
mae = mean_absolute_error(y_test, y_pred)
print("MAE for test data=",mae)

# %%
#sin関数をベースにノイズをのせたデータでの多項式回帰
from sklearn.preprocessing import PolynomialFeatures
import math
import random
import numpy as np
import matplotlib.pyplot as plt

xList = []
for i in range(0,51):
    xList.append(random.random()*math.pi*2)

yList_true = []
yList = []
for x in xList:
    yList.append(math.sin(x)+(random.random()*2.0-1.0))
    yList_true.append(math.sin(x))

fig4, ax4 = plt.subplots(figsize=(8,5),dpi=100)
ax4.scatter(xList,yList,label="train data")
ax4.scatter(xList,yList_true,label="sin curve")
ax4.legend()
fig4.savefig("fig4.png")

# %%
# sin関数をベースにノイズをのせたデータでの多項式回帰
from sklearn.linear_model import LinearRegression

pf = PolynomialFeatures(degree=3, include_bias = False)
x_train_poly = pf.fit_transform(np.array([xList]).T)

model = LinearRegression()
model.fit(x_train_poly, np.array([yList]).T)
print(model.coef_)

line = np.linspace(-0.5,6.5,200,endpoint=False).reshape(-1,1)
line_poly = pf.transform(line)

fig5, ax5 = plt.subplots(figsize=(8,5),dpi=100)
ax5.scatter(xList,yList,label="Train")
ax5.scatter(xList,yList_true,label="True")
ax5.plot(line, model.predict(line_poly), color='red' )
ax5.set_xlabel("x")
ax5.set_ylabel("y")
ax5.legend()
fig5.savefig("fig5.png")

# %%
pf = PolynomialFeatures(degree=3, include_bias = False)
x_train_poly = pf.fit_transform(np.array([xList]).T)

model = LinearRegression()
model.fit(x_train_poly, np.array([yList]).T)
print(model.coef_)

line = np.linspace(-0.5,6.5,200,endpoint=False).reshape(-1,1)
line_poly = pf.transform(line)

fig6, ax6 = plt.subplots(figsize=(8,5),dpi=100)
ax6.scatter(xList,yList,label="Train")
ax6.scatter(xList,yList_true,label="True")
ax6.plot(line, model.predict(line_poly), color='red' )
ax6.set_xlabel("x")
ax6.set_ylabel("y")
ax6.legend()
fig6.savefig("fig6.png")

# %%
#ラッソ回帰
from sklearn.linear_model import Lasso

pf = PolynomialFeatures(degree=8, include_bias = False)
x_train_poly = pf.fit_transform(np.array([xList]).T)

reg= Lasso(alpha=1.0)
reg.fit(x_train_poly, np.array([yList]).T)
print(reg.coef_)

line = np.linspace(-0.5,6.5,200,endpoint=False).reshape(-1,1)
line_poly = pf.transform(line)

fig7, ax7 = plt.subplots(figsize=(5,5),dpi=100)
ax7.scatter(xList,yList,label="Train")
ax7.scatter(xList,yList_true,label="True")
ax7.plot(line, reg.predict(line_poly), color='red' )
ax7.set_xlabel("x")
ax7.set_ylabel("y")
fig7.savefig("fig7.png")

# %%
#リッジ回帰
from sklearn.linear_model import Ridge

pf = PolynomialFeatures(degree=8, include_bias = False)
x_train_poly = pf.fit_transform(np.array([xList]).T)

reg= Ridge(alpha=1.0)
reg.fit(x_train_poly, np.array([yList]).T)
print(reg.coef_)

line = np.linspace(-0.5,6.5,200,endpoint=False).reshape(-1,1)
line_poly = pf.transform(line)

fig8, ax8 = plt.subplots(figsize=(5,5),dpi=100)
ax8.scatter(xList,yList,label="Train")
ax8.scatter(xList,yList_true,label="True")
ax8.plot(line, reg.predict(line_poly), color='red' )
ax8.set_xlabel("x")
ax8.set_ylabel("y")
fig8.savefig("fig8.png")

# %%
#ElasticNet

from sklearn.linear_model import ElasticNet

pf = PolynomialFeatures(degree=8, include_bias = False)
x_train_poly = pf.fit_transform(np.array([xList]).T)

#reg= ElasticNet(alpha=1.0, l1_ratio=0.5)
#ConvergenceWarningが出た場合、収束しないということなので、繰り返し回数: max_iterを増やすか、収束判定条件: tolを少し大きくする
reg= ElasticNet(alpha=1.0, l1_ratio=0.5,tol=0.01,max_iter=1000000)
reg.fit(x_train_poly, np.array([yList]).T)
print(reg.coef_)

line = np.linspace(0,6.5,200,endpoint=False).reshape(-1,1)
line_poly = pf.transform(line)

fig9, ax9 = plt.subplots(figsize=(5,5),dpi=100)
ax9.scatter(xList,yList,label="Train")
ax9.scatter(xList,yList_true,label="True")
ax9.plot(line, reg.predict(line_poly), color='red' )
ax9.set_xlabel("x")
ax9.set_ylabel("y")
fig9.savefig("fig9.png")

# %%

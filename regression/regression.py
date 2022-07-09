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
fig = plt.figure(figsize=(5,5),dpi=100)
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.scatter(x_train,y_train,label="Train")
plt.scatter(x_test,y_test,label="Test")
plt.plot([4,9],[4*a+b,9*a+b],color = "red")
plt.legend()
plt.show()

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
fig = plt.figure(figsize=(5,5),dpi=100)
plt.xlabel("actual value")
plt.ylabel("expected value")
plt.scatter(y_test,y_pred)
plt.plot([0,50],[0,50],color="red",ls="--") #一致した場合にプロットが乗る直線
plt.show()

# %%

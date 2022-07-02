import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

column_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv']
df=pd.read_csv("Boston.csv")

for col in column_names:
    if df[col].count()!=506:
        df[col].fillna(df[col].mode(),inplace=True)

#Dropping CHAS because it has discrete values
df=df.drop('chas',axis=1)

# Box plot diagram

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.boxplot(y=k, data=df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()


#Outliers percentage in every column

for k, v in df.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))


# Column Unnamed: 0 outliers = 0.00%
# Column crim outliers = 13.04%
# Column zn outliers = 13.44%
# Column indus outliers = 0.00%
# Column nox outliers = 0.00%
# Column rm outliers = 5.93%
# Column age outliers = 0.00%
# Column dis outliers = 0.99%
# Column rad outliers = 0.00%
# Column tax outliers = 0.00%
# Column ptratio outliers = 2.96%
# Column black outliers = 15.22%
# Column lstat outliers = 1.38%
# Column medv outliers = 7.91%


#Dropping Outliers greater than 10%
df=df.drop('crim',axis=1)
df=df.drop('zn',axis=1)
df=df.drop('black',axis=1)

# MEDV max value is greater than 50.Based on that, values above 50.00 may not help to predict MEDV
# lets remove MEDV value above 50
df= df[~(df['medv'] >= 50.0)]

#Instances
lreg=LinearRegression()
rf=RandomForestRegressor(random_state=0)
gb=GradientBoostingRegressor(n_estimators=10)
dc=DecisionTreeRegressor(random_state=0)
sv=SVR()
mlp=MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)
gnb=GaussianNB()
mnb=MultinomialNB()

#TRAINING AND TESTING
x=df.drop('medv',axis=1)
y=df['medv']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
lreg.fit(x_train,y_train)
y_pred=lreg.predict(x_test)
print("Linear REGRESSION",mean_squared_error(y_test, y_pred))

rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print("RANDOM FOREST",mean_squared_error(y_test, y_pred))
#
gb.fit(x_train,y_train)
y_pred=gb.predict(x_test)
print("GRADIENT BOOSTING",mean_squared_error(y_test, y_pred))
#
dc.fit(x_train,y_train)
y_pred=dc.predict(x_test)
print("DECISION TREE",mean_squared_error(y_test, y_pred))
#
sv.fit(x_train,y_train)
y_pred=sv.predict(x_test)
print("SVR",mean_squared_error(y_test, y_pred))
#
mlp.fit(x_train,y_train)
y_pred=mlp.predict(x_test)
print("MLP",mean_squared_error(y_test, y_pred))
#
#

# Linear REGRESSION 20.022579296721258
# RANDOM FOREST 13.12861101020406
# GRADIENT BOOSTING 23.993517766598472
# DECISION TREE 24.24030612244898
# SVR 54.618221352778626
# MLP 21.941391993651763
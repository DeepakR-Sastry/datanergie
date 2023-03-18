from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

df = pd.read_csv("data.csv")
X = df.drop('power_draw', axis=1)
y = df[["power_draw"]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=0.15)



regressor = RandomForestRegressor(n_estimators = 100, criterion="squared_error")
regressor.fit(X_train, y_train.values.ravel())
print()
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_pred, y_test)
print(mse)
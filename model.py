from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

df = pd.read_csv("dataA100.csv")
X = df.drop(['power_draw', 'env'], axis=1)
y = df["power_draw"]
env = df["env"]

mse_scores = []

unique_envs = env.unique()

for test_env in unique_envs:
    # Filter data for training on all envs except the test_env
    X_train = X[env != test_env]
    y_train = y[env != test_env]

    # Filter data for testing on the test_env
    X_test = X[env == test_env]
    y_test = y[env == test_env]

    regressor = RandomForestRegressor(n_estimators=100, criterion="mse")
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_pred, y_test)
    mse_scores.append(mse)

average_mse = np.mean(mse_scores)
print("Average Mean Squared Error:", average_mse)

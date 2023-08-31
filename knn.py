import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import numpy as np

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_elements = y_true != 0  # To avoid division by zero
    return np.mean(np.abs((y_true[non_zero_elements] - y_pred[non_zero_elements]) / y_true[non_zero_elements])) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))) * 100

# Read the CSV into a pandas DataFrame
df = pd.read_csv("trainDataA100.csv")
unique_envs = df['env'].unique()
total_rmse = 0
total_mae = 0
total_mape = 0
total_smape = 0
num_envs = len(unique_envs)

# Separate by environments
for env in unique_envs:
    # Split the data into training and test sets
    train_df = df[df['env'] != env]
    test_df = df[df['env'] == env]
    
    # Prepare training and test data
    X_train = train_df[['num_envs', 'num_actors', 'num_dofs', 'num_bodies']]
    y_train = train_df['energy']
    X_test = test_df[['num_envs', 'num_actors', 'num_dofs', 'num_bodies']]
    y_test = test_df['energy']

    # Scale the features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the KNN model
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = knn.predict(X_test_scaled)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_test, y_pred)
    print(f"Evaluation on held-out environment '{env}':")
    print(f"RMSE = {rmse}")
    print(f"MAE = {mae}")
    print(f"MAPE = {mape}%")
    print(f"SMAPE = {smape}%")


    total_rmse += rmse
    total_mae += mae
    total_mape += mape
    total_smape += smape


average_rmse = total_rmse / num_envs
average_mae = total_mae / num_envs
average_mape = total_mape / num_envs
average_smape = total_smape / num_envs

print("Average RMSE:", average_rmse)
print("Average MAE:", average_mae)
print("Average MAPE:", average_mape)
print("Average SMAPE:", average_smape)


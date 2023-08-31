import pandas as pd
import numpy as np  # Numpy is required for some mathematical operations
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE formula

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred))/2))) * 100


# Read the CSV into a pandas DataFrame
df = pd.read_csv("trainDataA100.csv")

# Separate by environments
unique_envs = df['env'].unique()
total_rmse = 0
total_mae = 0
total_mape = 0  # Initialize sum of MAPEs
total_smape = 0
num_envs = len(unique_envs)

for env in unique_envs:
    train_df = df[df['env'] != env]
    test_df = df[df['env'] == env]
    
    X_train = train_df[['num_envs', 'num_actors', 'num_dofs', 'num_bodies']]
    y_train = train_df['energy']
    X_test = test_df[['num_envs', 'num_actors', 'num_dofs', 'num_bodies']]
    y_test = test_df['energy']
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    y_pred = rf.predict(X_test_scaled)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)  # Calculate MAPE
    smape = symmetric_mean_absolute_percentage_error(y_test, y_pred)
    print(f"Evaluation on held-out environment '{env}': RMSE = {rmse}, MAE = {mae}, MAPE = {mape}%, SMAPE = {smape}%")
    
    total_rmse += rmse
    total_mae += mae
    total_mape += mape
    total_smape += smape

average_rmse = total_rmse / num_envs
average_mae = total_mae / num_envs
average_mape = total_mape / num_envs
average_smape = total_smape / num_envs
print(f"Average RMSE over all held-out environments: {average_rmse}")
print(f"Average MAE over all held-out environments: {average_mae}")
print(f"Average MAPE over all held-out environments: {average_mape}%")  # Report the average MAPE
print(f"Average SMAPE over all held-out environments: {average_smape}%")
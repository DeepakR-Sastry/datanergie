import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Read the CSV into a pandas DataFrame
df = pd.read_csv("simDataA100.csv")  # replace 'your_file.csv' with your actual filename

# Separate by environments
unique_envs = df['env'].unique()
total_rmse = 0  # Initialize sum of RMSEs
total_mae = 0  # Initialize sum of MAEs
num_envs = len(unique_envs)  # Number of unique environments

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
    scaler.fit(X_train)  # fit only on training data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # use the same scaling as training data

    # Train the Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = rf.predict(X_test_scaled)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Evaluation on held-out environment '{env}': RMSE = {rmse}, MAE = {mae}")
    
    # Add this RMSE and MAE to the running total
    total_rmse += rmse
    total_mae += mae

# Calculate the average RMSE and MAE over all held-out environments
average_rmse = total_rmse / num_envs
average_mae = total_mae / num_envs
print(f"Average RMSE over all held-out environments: {average_rmse}")
print(f"Average MAE over all held-out environments: {average_mae}")

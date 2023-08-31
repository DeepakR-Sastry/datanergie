import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('trainDataA100.csv')

# Drop the 'env' column, as we only want numerical columns for correlation
df = df.drop(['env'], axis=1)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Extract the 'energy' row from the correlation matrix
energy_correlation = correlation_matrix.loc['energy']

# Remove the 'energy-energy' correlation (it's always 1)
energy_correlation = energy_correlation.drop('energy')

# Generate a bar plot using Matplotlib
plt.figure(figsize=(10, 6))
energy_correlation.plot(kind='bar', color='blue')
plt.title('Correlation of Energy with Other Variables')
plt.xlabel('Variables')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=0)

# Save the plot as an image file
plt.savefig('energy_correlation.png')

print("Correlation of energy with other variables has been saved as 'energy_correlation.png'")

import pandas as pd
import numpy as np

# Load the dataset
file_path = 'customer_spending_data.csv'  # Make sure to use the correct file path
data = pd.read_csv(file_path)

# Scatter the 'Spending on Food' and 'Spending on Clothing' values slightly
# Apply random noise to scatter the values within a small range to make the data less clustered
np.random.seed(42)  # For reproducibility
scattering_factor = 5  # Adjust the scattering factor if needed

data['Spending on Food'] += np.random.uniform(-scattering_factor, scattering_factor, size=len(data))
data['Spending on Clothing'] += np.random.uniform(-scattering_factor, scattering_factor, size=len(data))

# Save the modified dataset with scattered values as a new CSV file
output_file_path = 'customer_spending_scattered_data.csv'  # New file path for the modified data
data.to_csv(output_file_path, index=False)

print(f"Modified dataset saved as: {output_file_path}")

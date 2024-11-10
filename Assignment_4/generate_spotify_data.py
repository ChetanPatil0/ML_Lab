import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 300

# Generate random data
data = {
    'Region': np.random.choice(['US', 'IN', 'UK', 'CA', 'AU'], num_samples),
    'Artist': [f'Artist{np.random.randint(1, 21)}' for _ in range(num_samples)],  # Random artists from Artist1 to Artist20
    'Song Length (min)': np.round(np.random.uniform(2.0, 5.0, num_samples), 2),  # Length between 2.0 and 5.0 minutes
    'Language': np.random.choice(['English', 'Hindi'], num_samples),
    'Genre': np.random.choice(['Pop', 'Rock', 'Bollywood', 'Jazz', 'Hip-Hop'], num_samples),
    'Release Year': np.random.choice(range(2000, 2024), num_samples),  # Years from 2000 to 2023
    'Popularity Score': np.random.randint(50, 100, num_samples)  # Popularity score between 50 and 100
}

# Create a DataFrame
dataset = pd.DataFrame(data)

# Save the DataFrame as a CSV file
dataset.to_csv('spotify_2023_data_extended.csv', index=False)

print("CSV file 'spotify_2023_data_extended.csv' with 300 rows created successfully.")

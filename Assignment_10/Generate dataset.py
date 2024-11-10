import pandas as pd
import random

# Function to generate random page visits (1 = visited, 0 = not visited)
def generate_page_visits(num_sessions, num_pages):
    data = []
    for session in range(1, num_sessions + 1):
        session_data = [f"Session_{session}"]
        session_data += [random.choice([0, 1]) for _ in range(num_pages)]
        data.append(session_data)
    return data

# Parameters
num_sessions = 70  # Number of sessions (rows)
num_pages = 10  # Number of web pages (columns)

# Generate the dataset
columns = ['Session_ID'] + [f'Page_{chr(65+i)}' for i in range(num_pages)]  # Pages named 'Page_A', 'Page_B', ...
data = generate_page_visits(num_sessions, num_pages)

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
file_path = r'C:\Users\Admin\Desktop\ML\Assignment_10\web_log_dataset_generated.csv'
df.to_csv(file_path, index=False)

print(f"Dataset generated and saved to {file_path}")

import pandas as pd
import numpy as np

# Number of rows
rows = 500

# Generating hours spent driving (random values between 0 and 12)
np.random.seed(42)  # For reproducibility
hours_spent_driving = np.random.uniform(0, 12, rows)

# Risk of backache (a function of hours, with some random variation)
backache_risk = np.clip(hours_spent_driving / 12 + np.random.normal(0, 0.1, rows), 0, 1)

# Creating a DataFrame
data = pd.DataFrame({
    'hours_spent_driving': hours_spent_driving,
    'backache_risk': backache_risk
})

# Show the first 5 rows of the generated dataset
data.head()

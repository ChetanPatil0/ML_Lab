import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df = pd.read_csv('web_log_dataset_generated.csv')

# Drop Session_ID since it's not relevant for the algorithm
df_apriori = df.drop('Session_ID', axis=1)

# Convert the integer columns (0, 1) to boolean (True, False)
df_apriori = df_apriori.astype(bool)

# Apply the Apriori algorithm with a minimum support threshold of 0.3
frequent_itemsets = apriori(df_apriori, min_support=0.3, use_colnames=True)

# Generate association rules with a minimum confidence threshold of 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Adjust display options to show all rows
pd.set_option('display.max_rows', None)  # Allow showing all rows
pd.set_option('display.max_columns', None)  # Allow showing all columns

# Display the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)

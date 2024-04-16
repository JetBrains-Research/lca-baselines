import os
import pandas as pd

# specify your directory path
directory = "/Users/Maria.Tigina/PycharmProjects/lca-baselines/data/upd"

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        df = df.drop(columns=['diff', 'issue_body'], errors='ignore')
        df.to_csv(filepath, index=False)

print("Columns deleted successfully from all CSV files.")

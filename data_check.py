import pandas as pd
df = pd.read_csv("artifacts/data.csv")
print("✅ Data Columns in Artifacts:", df.columns)
print("✅ Sample Data:\n", df.head())

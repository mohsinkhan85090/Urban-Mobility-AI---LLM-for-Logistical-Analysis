import pandas as pd

df = pd.read_csv("data/merged_taxi_data.csv")
print(df.columns.tolist())
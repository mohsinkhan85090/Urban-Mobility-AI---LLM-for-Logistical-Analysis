import pandas as pd
from config import CSV_PATH

df = pd.read_csv(CSV_PATH)
df1 = df.head(2000)
df1.to_csv('output.csv', index=False)
print('Ban gyi')
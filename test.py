import pandas as pd
df = pd.read_parquet('data/processed/hourly_features')
g = df[df.substation=='Avondale 33_11kV'].load_mw
print(g.describe())
print("hours below 0.1 MW:", (g < 0.1).sum())
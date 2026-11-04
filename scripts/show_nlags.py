'''Measures how strongly Hurstville’s demand resembles its own past at specific time lags, known as autocorrelation'''

import pandas as pd
from statsmodels.tsa.stattools import acf
df = pd.read_parquet("data/processed/hourly_features")
s = df[df["substation"].str.contains("Hurstville")].sort_values("timestamp")["load_mw"]
a = acf(s.dropna().to_numpy(), nlags=200)
for k in (24, 48, 144, 168, 192):
    print(k, round(float(a[k]), 3))
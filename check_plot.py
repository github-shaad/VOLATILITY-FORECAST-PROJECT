import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_parquet("results/har_1min_100_scaled.parquet")
pred = df["HAR_PRED"]
original = df["ACTUAL_RV"]
scaled_original = original.ewm(span=10,adjust=False).mean()
upper_limit = original.quantile(0.95)
clipped_original = original.clip(upper=upper_limit)
plt.figure(figsize=(14,6))
plt.plot(df.index, original, color="#9E2020")
plt.plot(df.index, pred, color="#4B2194")
plt.show()
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox


df = pd.read_parquet("results/har_1min_100_scaled.parquet")

y_hat = df["PRED"]
y = df["ACTUAL_RV"]
resid = y - y_hat

res = acorr_ljungbox(resid, lags = [12,78], model_df=3)
print(res)

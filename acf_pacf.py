import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


df = pd.read_parquet("data/realized_volatility_1min.parquet")

data = df["RV_1MIN"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(data, ax=axes[0], lags=40)
axes[0].set_title("Autocorrelation (ACF)")

plot_pacf(data, ax=axes[1], lags=40)
axes[1].set_title("Partial Autocorrelation (PACF)")

plt.show()



print(df.head(5))
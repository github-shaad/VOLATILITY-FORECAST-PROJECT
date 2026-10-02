import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("results/parameters/figarch_5min_100_scaled.csv")

param = "q"
paramls= df[param]
sns.histplot(paramls, bins=20, color="#1ABB0B")

plt.title(f"Parameter distribution plot for {param}")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()
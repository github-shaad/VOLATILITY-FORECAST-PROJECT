import numpy as np
import pandas as pd


df = pd.read_parquet("data/nq_100ms_data.parquet")

print(df.head(5))

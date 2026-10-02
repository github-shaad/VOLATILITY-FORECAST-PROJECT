import numpy as np
import pandas as pd
from dieboldmariano import dm_test

# 1. Load the results
df_3min = pd.read_parquet("results/figarch_1min_100_scaled.parquet")
df_1min = pd.read_parquet("results/har_1min_100_scaled.parquet")


# 2. Resample the 1-minute forecasts to 3-minute buckets
df_1min_agg = df_1min.resample("1 min").sum()

# 3. Force exact alignment using an inner join on the timestamp index
aligned_df = pd.merge(
    df_3min, 
    df_1min_agg[['PRED']], 
    left_index=True, 
    right_index=True, 
    suffixes=('', '_1MIN_AGG')
)

print(f"Aligned Length: {len(aligned_df)}")

# 4. Extract as NumPy arrays for the DM test
actual = aligned_df["ACTUAL_RV"].to_numpy()
pred_3min = aligned_df["PRED"].to_numpy()
pred_1min_agg = aligned_df["PRED_1MIN_AGG"].to_numpy()

# 5. Run the Diebold-Mariano Test
result = dm_test(actual, pred_3min, pred_1min_agg, one_sided=True)
print(result)
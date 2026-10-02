from arfima import ARFIMA, FIGARCH
import yaml
from helper_functions import filter_model_type, prepare_data
import pandas as pd
import numpy as np
from pprint import pprint
import gc
import os


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

active_runs = filter_model_type(config, "FIGARCH")
results_dir = config["data"]["results_dir"]
df_100ms = pd.read_parquet("data/nq_100ms_data.parquet")

def gen_features(model_data, rv_col, rv_scale, variance_floor=1e-8, use_log=False):
    """
    Sanitizes and prepares the univariate Realized Variance target for FIGARCH.
    """
    df_features = model_data.copy()
    scaled_rv = df_features[rv_col] * (rv_scale * rv_scale)
  
    df_features["FIGARCH_TARGET"] = scaled_rv.clip(lower=variance_floor)
    

    if use_log:
        df_features["FIGARCH_TARGET"] = np.log(df_features["FIGARCH_TARGET"])
        
  
    df_features = df_features.dropna()
 
    return df_features[["RESAMPLED_RETURNS", "FIGARCH_TARGET", rv_col]]


for run_name in active_runs:
    run_profile = config["experiments"][run_name]
    pprint(run_profile)
    freq = run_profile["resample_freq"]
    return_scale = run_profile["return_scale"]
    use_log = run_profile["use_log"]
    rv_col = f"RV_{freq.upper()}"
    rv_path = f"data/realized_volatility_{freq}.parquet"

    base_data = prepare_data(df_100ms, freq, rv_path)
    design_matrix = gen_features(base_data, rv_col, return_scale, use_log=use_log)

    train_batch_size = run_profile["train_batch_size"]
    test_batch_size = run_profile["test_batch_size"]

    p_list = run_profile["p_list"]
    q_list = run_profile["q_list"]

    i = train_batch_size
    preds = []
    param_list = []
    print(i, len(design_matrix), test_batch_size)
    while i <= len(design_matrix) - test_batch_size:
        train_start = i - train_batch_size
        train_end = i
        test_end = i + test_batch_size

        train_data = design_matrix.iloc[train_start: train_end]
        test_data = design_matrix.iloc[train_end: test_end]
        
        train_y = train_data["FIGARCH_TARGET"].to_numpy()

        figarch_model = FIGARCH()

        figarch_model.fit(squared_residuals=train_y, p_list=p_list, q_list=q_list)
        param_list.append({"p":figarch_model.p, "q":figarch_model.q, "d": figarch_model.d_v})
        forecasted_variance = figarch_model.predict(steps=test_batch_size)
        if use_log:
            # Convert both the forecast AND the historical max back to normal space
            forecasted_variance = np.exp(forecasted_variance)
            max_historical_var = np.max(np.exp(train_y)) * 3.0
        else:
            max_historical_var = np.max(train_y) * 3.0
        forecasted_variance = np.clip(forecasted_variance, a_min=1e-8, a_max=max_historical_var)

        forecasted_variance = forecasted_variance / (return_scale*return_scale)

        for step in range(test_batch_size):
            preds.append({
                'TIME_STAMP': test_data.index[step],
                'ACTUAL_RV': test_data[rv_col].iloc[step],
                'PRED': forecasted_variance[step]
            })
        param_list.append({"p":figarch_model.p, "q":figarch_model.q, "d": figarch_model.d_v})
        i += test_batch_size

   
    results_df = pd.DataFrame(preds).set_index('TIME_STAMP')
    results_path = os.path.join(results_dir, f"{run_name}.parquet")
    results_df.to_parquet(results_path)
    print(results_df.head())

    params_df = pd.DataFrame(param_list)
    params_df.to_csv(f"results/parameters/{run_name}.csv")
    del base_data, design_matrix
    gc.collect()




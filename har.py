import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
import gc
from pprint import pprint
from helper_functions import filter_model_type
"""
NEXT - DO :
1. GARCH
2. PLOTS
3. METRICS
4. EXPERIMENT LOG SYSTEM
"""

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

active_runs = filter_model_type(config, "HAR")
results_dir = config["data"]["results_dir"]
df_100ms = pd.read_parquet("data/nq_100ms_data.parquet")

def prepare_data(df, freq, rv_path):
    returns_resampled = df["LOG_RETURNS"].resample(freq).sum().rename("RESAMPLED_RETURNS")
    rv = pd.read_parquet(rv_path)[f"RV_{freq.upper()}"]

    model_data = pd.concat([returns_resampled, rv], axis=1, join="inner")
    model_data = model_data.between_time("09:30", "16:00")

    model_data = model_data.dropna()

    return model_data

def gen_features(model_data, lags, rv_col, rv_scale):
    df_features = model_data.copy()

    for lag in lags:
        scaled_rv = df_features[rv_col] * (rv_scale*rv_scale)
        df_features[f"RV_LAG_{lag}"] = scaled_rv.shift(1).rolling(window=lag).mean()

    df_features = df_features.dropna()

    return df_features


for run_name in active_runs:
    run_profile = config["experiments"][run_name]
    pprint(run_profile)
    freq = run_profile["resample_freq"]
    lags = run_profile["har_lags"]
    return_scale = run_profile["return_scale"]
    lag_names = [f"RV_LAG_{s}" for s in lags]
    
    rv_col = f"RV_{freq.upper()}"
    rv_path = f"data/realized_volatility_{freq}.parquet"

    base_data = prepare_data(df_100ms, freq, rv_path)
    design_matrix = gen_features(base_data, lags, rv_col, return_scale)
    returns = df_100ms["LOG_RETURNS"].resample(freq).sum().rename("LOG_RETURNS")

    train_batch_size = run_profile["train_batch_size"]
    test_batch_size = run_profile["test_batch_size"]

    i = train_batch_size
    preds = []

    while i <= len(design_matrix) - test_batch_size:
        train_start = i - train_batch_size
        train_end = i
        test_end = i + test_batch_size

        train_data = design_matrix.iloc[train_start: train_end]
        test_data = design_matrix.iloc[train_end: test_end]

        train_X = sm.add_constant(train_data[lag_names])
        train_y = train_data[rv_col]

        har_model = sm.OLS(train_y, train_X).fit()
        
        test_X = sm.add_constant(test_data[lag_names], has_constant='add')
        test_y = test_data[rv_col]

        har_pred = har_model.predict(test_X)
        har_pred = har_pred / (return_scale*return_scale)
        batch_preds = pd.DataFrame({
            'TIME_STAMP': test_data.index,
            'ACTUAL_RV': test_data[rv_col].to_numpy(dtype=float),
            'PRED': har_model.predict(test_X).to_numpy(dtype=float)
        })
        preds.append(batch_preds)

        i += test_batch_size

    results_df = pd.concat(preds, ignore_index=True).set_index('TIME_STAMP')
    results_path = results_dir + run_name + ".parquet"
    results_df.to_parquet(results_path)
    print(results_df.head())

    params_df = pd.DataFrame(param_list)
    params_df.to_csv(f"results/parameters/{run_name}.csv")
    del base_data, design_matrix
    gc.collect()
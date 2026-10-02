import pandas as pd
import numpy as np
import yaml


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


active_runs = config["pipeline"]["active_runs"]

df = pd.read_parquet("nq_order_book.parquet")
datetime_str = df["Date"].astype(str) + df["Time"].astype(str)
df["TIME_STAMP"] = pd.to_datetime(datetime_str, format="%Y-%m-%d%H:%M:%S.%f")
df.set_index("TIME_STAMP", inplace=True)
df.sort_index(inplace=True)
df.drop(columns=["Date", "Time"], inplace=True)

df = df.groupby(level=0).last()

bid_price = df["BID_PRICE"]
ask_price = df["ASK_PRICE"]
bid_q = df["BID_SIZE"]
ask_q = df["ASK_SIZE"]
micro_price = (((bid_price*bid_q) + (ask_price*ask_q)) / (bid_q + ask_q)) 
df["MICRO_PRICE"] = micro_price
df.loc[df.index[0], "MICRO_PRICE"] = 0.0
df["RETURNS"] = df["MICRO_PRICE"].pct_change()
df["RETURNS"].replace([np.inf, -np.inf], np.nan, inplace=True)
log_prices = np.log(df["MICRO_PRICE"])
df["LOG_RETURNS"] = (log_prices - log_prices.shift(1))
df.dropna(subset=["LOG_RETURNS", "RETURNS", "MICRO_PRICE"], inplace=True)
df.to_parquet("data/nq_100ms_data.parquet")

df["SQ_LOG_RETURNS"] = df["LOG_RETURNS"] ** 2

for run_name in active_runs:
    run_profile = config["experiments"][run_name]
    freq = run_profile["resample_freq"]
    rv = df["SQ_LOG_RETURNS"].resample(freq).sum().rename(f"RV_{freq.upper()}").to_frame()
    rv_path = f"data/realized_volatility_{freq}.parquet"
    rv.to_parquet(rv_path)

print(df["MICRO_PRICE"].iloc[:5])



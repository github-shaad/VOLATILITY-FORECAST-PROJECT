import pandas as pd


def filter_model_type(config, model_type:str):
    active_runs = [i for i in config["pipeline"]["active_runs"] if config["experiments"][i]["model_type"] == model_type]
    return active_runs

def prepare_data(df, freq, rv_path):
    returns_resampled = df["LOG_RETURNS"].resample(freq).sum().rename("RESAMPLED_RETURNS")
    rv = pd.read_parquet(rv_path)[f"RV_{freq.upper()}"]

    model_data = pd.concat([returns_resampled, rv], axis=1, join="inner")
    model_data = model_data.between_time("09:30", "16:00")

    model_data = model_data.dropna()

    return model_data
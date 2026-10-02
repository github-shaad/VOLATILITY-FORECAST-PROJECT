import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yaml
from pprint import pprint
import statsmodels.api as sm
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'axes.edgecolor': '#333333',
})
def calculate_metrics(actual, predicted):
    """Calculates robust loss functions (MSE and QLIKE) for volatility evaluation."""
    # Drop NaNs to ensure clean calculation
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((actual - predicted) ** 2)
    
    # Calculate QLIKE with a microscopic epsilon to avoid math errors
    eps = 1e-10 
    qlike = np.mean((actual / (predicted + eps)) - np.log((actual + eps) / (predicted + eps)) - 1)

    hmse = np.mean(((actual / predicted) - 1)**2)
    X = sm.add_constant(predicted)
    mz_model = sm.OLS(actual, X).fit()
    mz_r2 = mz_model.rsquared
    return mse, qlike, hmse, mz_r2

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

active_runs = config["pipeline"]["active_runs"]
results_dir = config["data"]["results_dir"]
plots_dir = config["data"]["plots_dir"]
metrics_dir = config["data"]["metrics_dir"]
evaluation_results = []
for run_name in active_runs:
    run_profile = config["experiments"][run_name]
    pprint(run_profile)

    results_path = results_dir + run_name + ".parquet"
    results = pd.read_parquet(results_path)
    actual = results["ACTUAL_RV"]
    pred = results["PRED"]

    mse, qlike, hmse, mz_r2 = calculate_metrics(actual, pred)

    evaluation_results.append({'Model': run_name, 'MSE': mse, 'QLIKE-LOSS': qlike, "HMSE": hmse,
        "MZ_R2": mz_r2})

    fig, ax = plt.subplots(figsize=(14,6))
    
    # Format the title to look cleaner for the paper
    clean_title = run_name.replace("_", " ").upper()
    ax.set_title(f"Forecast vs Actual: {clean_title}", pad=15)
    
    # Swapped to standard academic colors (Blue/Red) with adjusted alphas for clarity
    ax.plot(results.index, actual, label="Actual Variance", color="#1f77b4", alpha=0.7, linewidth=1.2)
    ax.plot(results.index, pred, label="Forecasted Variance", color="#d62728", alpha=0.9, linewidth=1.5, linestyle="--")
    
    ax.set_ylabel("Variance")
    ax.set_xlabel("Time")
    
    # Add a shadow box to the legend so it pops off the grid
    ax.legend(loc="upper right", frameon=True, shadow=True)
    
    plt.tight_layout()
    plots_path = plots_dir + f"/{run_name}.png"
    plt.savefig(plots_path, dpi=300)
    
    # Memory management: clear and close the figure before the next run_name loop starts
    plt.clf()
    plt.close(fig)


new_evaluation_df = pd.DataFrame(evaluation_results).set_index(['Model'])
evaluation_path = metrics_dir + "/metrics.csv"

if os.path.exists(evaluation_path):
    # Load the existing data
    existing_df = pd.read_csv(evaluation_path).set_index('Model')
    
    # combine_first prioritizes new_evaluation_df, keeping old rows that aren't in the new one
    final_df = new_evaluation_df.combine_first(existing_df)
else:
    # If this is the very first run, just use the new data
    final_df = new_evaluation_df

# 3. Save the combined results
final_df.to_csv(evaluation_path)




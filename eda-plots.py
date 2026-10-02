import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


matplotlib.use('Agg')

# Set a clean, professional style for academic presentation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'lines.linewidth': 1.5,
    'axes.edgecolor': '#333333',
})

# 2. Load the Data
n_min = "3min"
rv = pd.read_parquet(f"data/realized_volatility_{n_min}.parquet")
original_df = pd.read_parquet("data/nq_100ms_data.parquet")

# 3. Create the directories if they do not exist
save_dir = f"data/eda/{n_min}"
os.makedirs(save_dir, exist_ok=True)

# 4. The memory-safe plotting loop
for s in range(3, 12):
    target_day = f"2025-11-{s:02d}"  # Zero-padding just in case (e.g., 03, 04)
    
    # Safely try to extract data for the day; if the weekend/holiday is empty, skip to avoid plotting errors
    try:
        rv_slice = rv.loc[target_day].between_time("09:30", "16:00")
        raw_slice = original_df.loc[target_day].between_time("09:30", "16:00")
        if rv_slice.empty or raw_slice.empty:
            print(f"Skipping {target_day}: No data found.")
            continue
    except KeyError:
        print(f"Skipping {target_day}: Date not in index.")
        continue

    # ==========================================
    # Plot 1: 1-Minute Realized Volatility
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(14,6))
    ax1.set_title(f"Intraday Realized Volatility {n_min} - {target_day}", pad=15)
    
    ax1.plot(rv_slice.index, rv_slice[f"RV_{n_min.upper()}"], color="#1f77b4", label="1-Min Realized Volatility")
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Time (EST)")
    
    ax1.legend(loc="upper left", frameon=True, shadow=True)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
    
    plt.tight_layout()
    
    # Save and aggressively close to free memory
    plt.savefig(f"{save_dir}/rv_{target_day}.png", dpi=300)
    plt.clf()     # Clear the current figure
    plt.close(fig) # Close the specific figure explicitly

    # ==========================================
    # Plot 2: 100ms Log Returns
    # ==========================================
    fig, ax2 = plt.subplots(figsize=(14,6))
    ax2.set_title(f"Log Returns (100ms) - {target_day}", pad=15)
    
    # Add alpha to slightly soften the dense scatter of 100ms returns
    ax2.plot(raw_slice.index, raw_slice["LOG_RETURNS"], color="#d62728", alpha=0.8, label="100ms Log Returns")
    ax2.set_ylabel("Log Return")
    ax2.set_xlabel("Time (EST)")
    
    ax2.legend(loc="upper right", frameon=True, shadow=True)
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
    
    plt.tight_layout()
    
    # Save and aggressively close to free memory
    plt.savefig(f"{save_dir}/lr_{target_day}.png", dpi=300)
    plt.clf()     # Clear the current figure
    plt.close(fig) # Close the specific figure explicitly

print("Batch plotting complete.")
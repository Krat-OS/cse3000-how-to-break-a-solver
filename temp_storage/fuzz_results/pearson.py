#!/usr/bin/env python3

import sys
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np

def main():
    # Path to your CSV file
    CSV_FILE = "s4100_combined_gpmc.csv"

    # 1) Read CSV
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: File '{CSV_FILE}' not found.")
        sys.exit(1)

    # 2) Verify required columns
    required_cols = ['solve_time', 'count_value']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in '{CSV_FILE}'.")
            sys.exit(1)

    # 3) Ignore rows where 'count_value' is empty (NaN)
    df.dropna(subset=['count_value'], inplace=True)

    # 4) Extract series (dropping any remaining NaN for time as well)
    time_series = df['solve_time'].dropna()
    count_series = df['count_value'].dropna()

    # 5) Restrict to common indices
    common_idx = time_series.index.intersection(count_series.index)
    time_vals = time_series.loc[common_idx]
    count_vals = count_series.loc[common_idx]

    # Take square root of count values for Pearson correlation
    sqrt_count_vals = np.power(count_vals, 1/4)

    # If no overlapping data, exit
    if len(time_vals) == 0:
        print("No overlapping data to compute correlation.")
        sys.exit(0)

    # 6) Pearson’s correlation (linear, with sqrt(count_value))
    pearson_coef_sqrt, pearson_p_sqrt = pearsonr(time_vals, sqrt_count_vals)
    print(f"Pearson correlation (with sqrt of count_value):  r = {pearson_coef_sqrt:.3f},  p-value = {pearson_p_sqrt:.3e}")

    # 7) Spearman’s rank correlation (monotonic)
    spearman_coef, spearman_p = spearmanr(time_vals, count_vals)
    print(f"Spearman correlation: r = {spearman_coef:.3f},  p-value = {spearman_p:.3e}")


if __name__ == "__main__":
    main()
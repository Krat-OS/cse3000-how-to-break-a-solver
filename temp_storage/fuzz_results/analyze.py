#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # -------------------------------------------------------------------------
    # 1) Parse command-line arguments
    # -------------------------------------------------------------------------
    if len(sys.argv) < 3:
        print("Usage: python analyze.py <solver_name> <feature_column> [-time] [-count]")
        print("       solver_name must be one of: gpmc, d4, ganak")
        sys.exit(1)

    solver_name = sys.argv[1]
    column_to_sort = sys.argv[2]

    if solver_name not in ['gpmc', 'd4', 'ganak']:
        print("Error: solver_name must be one of: gpmc, d4, ganak")
        sys.exit(1)

    # Determine which CSV file to read
    solver_to_csv = {
        'gpmc':  's4100_combined_gpmc.csv',
        'd4':    's4100_combined_d4.csv',
        'ganak': 's4100_combined_ganak.csv'
    }
    CSV_FILE = solver_to_csv[solver_name]

    # Parse flags
    plot_time = ('-time' in sys.argv)
    plot_count = ('-count' in sys.argv)

    # If no -time / -count provided, there's nothing to plot
    if not plot_time and not plot_count:
        print("Please specify at least one of -time or -count.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2) Read and sort the CSV
    # -------------------------------------------------------------------------
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: File '{CSV_FILE}' not found.")
        sys.exit(1)

    # Check if the requested sort column exists
    if column_to_sort not in df.columns:
        print(f"Error: Column '{column_to_sort}' not found in '{CSV_FILE}'.")
        sys.exit(1)

    # Sort by the feature column
    df.sort_values(by=column_to_sort, inplace=True)

    # Overwrite CSV with the sorted data
    df.to_csv(CSV_FILE, index=False)

    # -------------------------------------------------------------------------
    # 3) Prepare data for plotting
    # -------------------------------------------------------------------------
    # Convert x-axis column to numeric (if possible) to ensure we can do polyfit
    x_all = pd.to_numeric(df[column_to_sort], errors='coerce') * 100
    df = df[~x_all.isna()]
    x_all = x_all[~x_all.isna()]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # We'll keep track of lines and labels for a common legend
    lines  = []
    labels = []

    # Degree of the polynomial fit
    poly_degree = 20

    # -------------------------------------------------------------------------
    # 4) If -time flag is provided, plot time on the left y-axis
    # -------------------------------------------------------------------------
    if plot_time:
        # Ensure 'solve_time' is in the columns
        if 'solve_time' not in df.columns:
            print("Error: 'solve_time' column not found in CSV.")
            sys.exit(1)

        y_time_all = pd.to_numeric(df['solve_time'], errors='coerce')
        valid_mask_time = ~y_time_all.isna()
        x_time = x_all[valid_mask_time]
        y_time = y_time_all[valid_mask_time]

        # Scatter for time (size=2 dots)
        time_scatter = ax1.scatter(x_time, y_time, s=8, c='blue', alpha=0.6, label='Time')
        lines.append(time_scatter)
        labels.append('Time')

        # Polynomial fit for time
        if len(x_time) > poly_degree:  # Ensure enough points for the polynomial degree
            coeffs_time = np.polyfit(x_time, y_time, poly_degree)
            poly_time = np.poly1d(coeffs_time)
            x_dense_time = np.linspace(x_time.min(), x_time.max(), 2000)
            y_fit_time = poly_time(x_dense_time)
            time_fit_line, = ax1.plot(x_dense_time, y_fit_time, c='blue', linestyle='--', label='Time (polyfit)')
            # lines.append(time_fit_line)
            # labels.append('Time (polyfit)')

        ax1.set_xlabel("Horn Clauses Fraction", fontsize=22)
        ax1.set_ylabel("Time", fontsize=22)

    # -------------------------------------------------------------------------
    # 5) If -count flag is provided, plot count on the same or twin y-axis
    # -------------------------------------------------------------------------
    ax2 = None
    if plot_count:
        # Ensure 'count_value' is in the columns
        if 'count_value' not in df.columns:
            print("Error: 'count_value' column not found in CSV.")
            sys.exit(1)

        y_count_all = np.power(pd.to_numeric(df['count_value'], errors='coerce'), 1/4)
        valid_mask_count = ~y_count_all.isna()
        x_count = x_all[valid_mask_count]
        y_count = y_count_all[valid_mask_count]

        # If we already plotted time, use a twin axis on the right
        if plot_time:
            ax2 = ax1.twinx()
            ax_count = ax2
        else:
            ax_count = ax1

        # Scatter for count (size=3 dots)
        count_scatter = ax_count.scatter(x_count, y_count, s=8, c='red', alpha=0.6, label='Count')
        lines.append(count_scatter)
        labels.append('Count')

        # Polynomial fit for count
        if len(x_count) > poly_degree:  # Ensure enough points for the polynomial degree
            coeffs_count = np.polyfit(x_count, y_count, poly_degree)
            poly_count = np.poly1d(coeffs_count)
            x_dense_count = np.linspace(x_count.min(), x_count.max(), 2000)
            y_fit_count = poly_count(x_dense_count)
            count_fit_line, = ax_count.plot(x_dense_count, y_fit_count, c='red', linestyle='--', label='Count (polyfit)')
            # lines.append(count_fit_line)
            # labels.append('Count (polyfit)')

        if plot_time:
            ax_count.set_ylabel("Count", fontsize=22)
        else:
            ax_count.set_xlabel(column_to_sort, fontsize=22)
            ax_count.set_ylabel("Count", fontsize=22)

    # -------------------------------------------------------------------------
    # 6) Formatting: legend, no title, font sizes, etc.
    # -------------------------------------------------------------------------
    ax1.tick_params(axis='both', labelsize=14)  # for x axis & left y axis
    if ax2 is not None:
        ax2.tick_params(axis='y', labelsize=14)

    # Place a single legend on ax1
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='blue', markersize=10, label='Time'),
        plt.Line2D([0], [0], marker='o', color='red', markersize=10, label='Count')
    ]

    ax1.legend(handles=legend_handles, loc='best', fontsize=22, handletextpad=0.5)
    
    # No title
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
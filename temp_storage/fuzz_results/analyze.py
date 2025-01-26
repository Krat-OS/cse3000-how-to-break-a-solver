#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # -------------------------------------------------------------------------
    # 1) Parse command-line arguments
    # -------------------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <solver_name>")
        print("       solver_name must be one of: gpmc, d4, ganak")
        sys.exit(1)

    solver_name = sys.argv[1]

    if solver_name not in ['gpmc', 'd4', 'ganak']:
        print("Error: solver_name must be one of: gpmc, d4, ganak")
        sys.exit(1)

    # Determine which CSV file to read
    solver_to_csv = {
        'gpmc':  'brum_gpmc.csv',
        'd4':    'brum_d4.csv',
        'ganak': 'brum_ganak.csv'
    }
    CSV_FILE = solver_to_csv[solver_name]

    # -------------------------------------------------------------------------
    # 2) Read and clean the CSV
    # -------------------------------------------------------------------------
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: File '{CSV_FILE}' not found.")
        sys.exit(1)

    # Ensure required columns are in the CSV
    required_columns = ['count_value', 'solve_time']
    for column in required_columns:
        if column not in df.columns:
            print(f"Error: '{column}' column not found in '{CSV_FILE}'.")
            sys.exit(1)

    # Clean and convert columns to numeric
    df['count_value'] = pd.to_numeric(df['count_value'], errors='coerce')
    df['solve_time'] = pd.to_numeric(df['solve_time'], errors='coerce')

    # Remove rows with NaN values in required columns
    df.dropna(subset=['count_value', 'solve_time'], inplace=True)

    # -------------------------------------------------------------------------
    # 3) Plot the data
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 8))

    # Scatter plot
    scatter = ax.scatter(
        df['count_value'], 
        df['solve_time'], 
        s=8, 
        c='blue', 
        alpha=0.6, 
        label='Solve Time'
    )

    # Polynomial fit
    poly_degree = 2000
    if len(df) > poly_degree:  # Ensure enough points for the polynomial degree
        coeffs = np.polyfit(df['count_value'], df['solve_time'], poly_degree)
        poly = np.poly1d(coeffs)
        x_dense = np.linspace(df['count_value'].min(), df['count_value'].max(), 1000)
        y_fit = poly(x_dense)
        ax.plot(
            x_dense, 
            y_fit, 
            c='red', 
            linestyle='--', 
            label='Polynomial Fit'
        )

    # Labels and formatting
    ax.set_xlabel("Count Value", fontsize=22)
    ax.set_ylabel("Solve Time (seconds)", fontsize=22)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=18)

    # No title
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import sys
import pandas as pd

def calculate_average_solve_time(csv_file):
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Ensure the required columns exist
        required_columns = ['satisfiability', 'count_value', 'solve_time']
        for column in required_columns:
            if column not in df.columns:
                print(f"Error: Required column '{column}' not found in the CSV file.")
                sys.exit(1)

        # Filter rows that have a valid count_value or are UNSATISFIABLE
        valid_rows = df[(df['count_value'].notna()) | (df['satisfiability'] == 'UNSATISFIABLE')]

        # Calculate the average solve time
        if valid_rows.empty:
            print("No valid rows found for calculation.")
            return

        average_solve_time = valid_rows['solve_time'].mean()
        print(f"Average Solve Time: {average_solve_time:.6f} seconds")

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    calculate_average_solve_time(csv_file)
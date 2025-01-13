import sys
import os
import pandas as pd

def sort_file(file_path):
    # Read the CSV
    df = pd.read_csv(file_path)

    # Extract the two 3-digit numbers from the 'instance' column
    df['primary_num'] = df['instance'].str.extract(r'_(\d{3})_')[0].astype(int)  # First 3-digit number
    df['secondary_num'] = df['instance'].str.extract(r'_(\d{3}).cnf$')[0].astype(int)  # Second 3-digit number

    # Sort by secondary_num first, then by primary_num
    sorted_df = df.sort_values(by=['secondary_num', 'primary_num']).drop(columns=['primary_num', 'secondary_num'])

    # Save back to the same file
    sorted_df.to_csv(file_path, index=False)
    print(f"Sorted {file_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python sort.py <seed>")
        sys.exit(1)

    seed = sys.argv[1]
    directory = "/Users/vukjurisic/Work/Uni/RP/cse3000-how-to-break-a-solver/SharpVelvet/out/"

    # Get all files containing the given seed
    files = [f for f in os.listdir(directory) if seed in f and f.endswith(".csv")]

    if not files:
        print(f"No files found for seed: {seed}")
        sys.exit(1)

    # Sort each file
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        sort_file(file_path)

if __name__ == "__main__":
    main()
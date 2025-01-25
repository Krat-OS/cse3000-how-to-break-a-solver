import pandas as pd

def combine_csv_files(file1_path, file2_path, output_file_path):
    """
    Combine two CSV files based on the 'instance_name' column.
    Match by checking if the 'instance' in file1 contains the 'instance_name' in file2.

    Args:
        file1_path (str): Path to the first CSV file.
        file2_path (str): Path to the second CSV file.
        output_file_path (str): Path to save the combined CSV file.
    """
    try:
        # Read the first file
        print(f"Reading first file: {file1_path}")
        df1 = pd.read_csv(file1_path)
        print(f"First file read successfully with {len(df1)} rows.")

        # Read the second file
        print(f"Reading second file: {file2_path}")
        df2 = pd.read_csv(file2_path)
        print(f"Second file read successfully with {len(df2)} rows.")

        if 'instance_name' not in df2.columns:
            print("Error: 'instance_name' column not found in the second file.")
            return

        # Create a new column in file2 with ".cnf" appended to match file1's format
        df2['instance_name_with_cnf'] = df2['instance_name'] + '.cnf'

        # Filter rows in df1 where the 'instance' contains the 'instance_name_with_cnf' from df2
        print("Matching rows from file1 and file2...")
        matched_rows = df1[df1['instance'].str.contains('|'.join(df2['instance_name_with_cnf']), na=False)]

        # Merge the filtered rows from df1 with df2
        combined_df = pd.merge(matched_rows, df2, left_on=df1['instance'].apply(lambda x: x.split('/')[-1].replace('.cnf', '')),
                               right_on='instance_name', how='inner')

        print(f"Merged successfully with {len(combined_df)} rows.")

        # Save the combined DataFrame to a new CSV file
        combined_df.to_csv(output_file_path, index=False)
        print(f"Combined CSV file saved to: {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
file1_path = '/home/vjurisic/cse3000-how-to-break-a-solver/SharpVelvet/out/2025-01-19_chevu_000_s4090_000_d4g_fuzz-results.csv'
file2_path = '/home/vjurisic/cse3000-how-to-break-a-solver/SharpVelvet/out/features_output/3cnf-400clause-90var-horn_features_output.csv'
output_file_path = 's4090_combined_d4.csv'

combine_csv_files(file1_path, file2_path, output_file_path)

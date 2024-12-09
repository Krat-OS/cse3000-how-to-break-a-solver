import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import pandas as pd

from cpog_verifier.utils import run_command
from result_processor.utils import process_results


def run_python_script(script_path: str, **kwargs):
    """Run a Python script with optional arguments."""
    cmd = ["python3", script_path]
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    print(f"Running script: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def clear_and_create_directory(dir_path: str):
    """Clear the existing directory and recreate it."""
    if os.path.exists(dir_path):
        print(f"Deleting existing directory: {dir_path}")
        shutil.rmtree(dir_path)
    print(f"Creating directory: {dir_path}")
    os.makedirs(dir_path, exist_ok=True)

def merge_feature_csvs(csv_dir: Path) -> pd.DataFrame:
    """Merge all CSV files in directory into single DataFrame.

    Args:
        csv_dir: Directory containing CSV files to merge

    Returns:
        pd.DataFrame: Merged DataFrame containing all feature data
    """
    dfs: List[pd.DataFrame] = []

    for csv_file in csv_dir.glob("*.csv"):
        df = process_results(csv_file)
        df['instance_name'] = csv_file.stem
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No CSV files found in {csv_dir}")

    return pd.concat(dfs, ignore_index=True)

def compute_features(cnf_dir: str, features_output_dir: str, satzilla_path: str) -> None:
    """Compute features for all CNF files and store them in output directory.

    Args:
        cnf_dir: Directory containing CNF files
        features_output_dir: Directory to store output features
        satzilla_path: Path to SATzilla executable
    """
    cnf_dir_path = Path(cnf_dir)
    features_output_path = Path(features_output_dir)

    clear_and_create_directory(features_output_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for cnf_file in cnf_dir_path.glob("*.cnf"):
            output_file = temp_path / f"{cnf_file.stem}.csv"
            print(f"Computing features for: {cnf_file}")

            cmd = [satzilla_path, "-base", str(cnf_file), str(output_file)]
            return_code, _, stderr = run_command(cmd)

            if return_code != 0:
                print(f"Error computing features for {cnf_file}. Error: {stderr}")
                continue

            print(f"Features computed for: {cnf_file}")

        try:
            merged_df = merge_feature_csvs(temp_path)

            output_file = features_output_path / "features_output.csv"
            merged_df.to_csv(output_file, index=False)
            print(f"Merged features saved to: {output_file}")

        except ValueError as e:
            print(f"Error merging CSV files: {e}")

def process_csv_files(features_output_dir: str):
    """Process CSV files to remove duplicates."""
    for csv_file in os.listdir(features_output_dir):
        if csv_file.endswith(".csv"):
            csv_file_path = os.path.join(features_output_dir, csv_file)
            try:
                print(f"Processing CSV file: {csv_file_path}")
                df = pd.read_csv(csv_file_path, header=None)
                first_row = df.iloc[0]
                df = df[df.ne(first_row).any(axis=1)]
                df.iloc[0] = first_row
                df.to_csv(csv_file_path, index=False, header=False)
                print(f"Processed successfully: {csv_file_path}")
            except Exception as e:
                print(f"Error processing {csv_file_path}: {e}")

def delete_old_instances(directory: str):
    """Delete old instances directory if it exists."""
    if os.path.exists(directory):
        print(f"Deleting existing instances directory: {directory}")
        shutil.rmtree(directory)
    else:
        print(f"No existing instances directory found at: {directory}")

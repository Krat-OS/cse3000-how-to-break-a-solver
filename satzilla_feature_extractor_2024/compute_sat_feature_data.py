import os
import shutil
import subprocess
import pandas as pd

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
        os.system(f"rm -rf {dir_path}")
    print(f"Creating directory: {dir_path}")
    os.makedirs(dir_path, exist_ok=True)


def compute_features(cnf_dir: str, features_output_dir: str, satzilla_path: str):
    """Compute features for all CNF files."""
    clear_and_create_directory(features_output_dir)
    for cnf_file in os.listdir(cnf_dir):
        if cnf_file.endswith(".cnf"):
            generator_type = cnf_file.split("_")[0]
            output_file = os.path.join(features_output_dir, f"features_output_{generator_type}.csv")
            print(f"Computing features for: {cnf_file}")
            subprocess.run([satzilla_path, "-base", os.path.join(cnf_dir, cnf_file), output_file], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Features saved to: {output_file}")



def process_csv_files(features_output_dir: str):
    """
    Process CSV files to remove duplicates and display their full paths.

    Args:
        features_output_dir (str): Path to the directory containing CSV files.
    """
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
                print(f"Error processing {csv_file_path}: {e}")\


def delete_old_instances(directory: str):
    """
    Deletes the old instances directory if it exists.
    
    Args:
        directory (str): Path to the instances directory.
    
    Returns:
        None
    """
    if os.path.exists(directory):
        print(f"Deleting existing instances directory: {directory}")
        shutil.rmtree(directory)
    else:
        print(f"No existing instances directory found at: {directory}")
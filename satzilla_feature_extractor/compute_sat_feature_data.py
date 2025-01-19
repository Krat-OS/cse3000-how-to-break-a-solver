import os
import re
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from cpog_verifier.utils import run_command
from result_processor.utils import process_results

###############################################################################
# Logging Setup
###############################################################################
class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter that adds contextual information to log messages.

    The formatter displays:
    - Level name only for WARNING and ERROR levels
    - Timestamp and message content
    """


    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record.

        Args:
            record: The log record to format

        Returns:
            Formatted log message string
        """

        return (
            f"[GLOBAL CLI {record.levelname}], "
            f"{self.formatTime(record)}: "
            f"{record.getMessage()}"
        )

satzilla_logger = logging.getLogger("SATZilla Feature Extractor Tool")
satzilla_logger.setLevel(logging.INFO)

local_handler = logging.StreamHandler()
local_handler.setFormatter(CustomFormatter())
satzilla_logger.addHandler(local_handler)

satzilla_logger.propagate = False

def run_python_script(script_path: str, **kwargs) -> None:
    """
    Run a Python script with optional arguments.

    Args:
        script_path (str): Path to the Python script to execute.
        kwargs: Additional key-value arguments to pass to the script.
    """
    cmd = ["python3", script_path]
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    satzilla_logger.info(f"Running script: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def merge_feature_csvs(csv_dir: Path) -> pd.DataFrame:
    """
    Merge all CSV files in directory into single DataFrame.

    Args:
        csv_dir (Path): Directory containing CSV files to merge.

    Returns:
        pd.DataFrame: Merged DataFrame containing all feature data.
    """
    satzilla_logger.info(f"Merging feature CSV files from directory: {csv_dir}")
    dfs: List[pd.DataFrame] = []

    for csv_file in csv_dir.glob("*.csv"):
        satzilla_logger.debug(f"Processing file: {csv_file}")
        df = process_results(csv_file)
        df["instance_name"] = csv_file.stem
        dfs.append(df)

    if not dfs:
        satzilla_logger.error(f"No CSV files found in {csv_dir}")
        raise ValueError(f"No CSV files found in {csv_dir}")

    merged_df = pd.concat(dfs, ignore_index=True)
    satzilla_logger.info("Feature CSV files merged successfully.")
    return merged_df


def extract_generator_seed_and_add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract generator and seed from the 'instance_name' column and add them as new columns.

    Args:
        df (pd.DataFrame): DataFrame containing the merged feature data with 'instance_name' column.

    Returns:
        pd.DataFrame: DataFrame with added 'generator' and 'seed' columns.
    """
    satzilla_logger.info("Extracting generator and seed from 'instance_name'.")
    if "instance_name" not in df.columns:
        satzilla_logger.error("'instance_name' column is missing from the DataFrame.")
        raise ValueError("'instance_name' column is missing from the DataFrame.")

    def extract_generator_and_seed(instance_name: str) -> Dict[str, Optional[object]]:
        pattern: re.Pattern = re.compile(r"(?P<generator>.+)_(?:\d+)_s(?P<seed>\d+)$")
        match: Optional[re.Match] = pattern.match(instance_name)
        if match:
            return {
                "generator": match.group("generator"),
                "seed": int(match.group("seed")),
            }
        return {"generator": None, "seed": None}

    extracted_data: pd.Series = df["instance_name"].apply(extract_generator_and_seed)
    extracted_df: pd.DataFrame = pd.DataFrame(list(extracted_data))
    enhanced_df = pd.concat([df, extracted_df], axis=1)
    satzilla_logger.info("Generator and seed extraction completed.")
    return enhanced_df


def compute_features(cnf_dir: str, features_output_dir: str, satzilla_path: str) -> None:
    """
    Compute features for all CNF files and store them in output directory.

    Args:
        cnf_dir (str): Directory containing CNF files.
        features_output_dir (str): Directory to store output features.
        satzilla_path (str): Path to SATzilla executable.
    """
    cnf_dir_path: Path = Path(cnf_dir)
    features_output_path: Path = Path(features_output_dir)

    satzilla_logger.info(f"Starting feature extraction for CNFs in: {cnf_dir_path}")

    if not cnf_dir_path.is_dir():
        satzilla_logger.error(f"The specified CNF directory does not exist: {cnf_dir}")
        raise FileNotFoundError(f"The specified CNF directory does not exist: {cnf_dir}")

    output_stem: str = cnf_dir_path.parent.parent.name
    output_file_name: str = f"{output_stem}_features_output.csv"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path: Path = Path(temp_dir)

        for cnf_file in cnf_dir_path.glob("*.cnf"):
            output_file: Path = temp_path / f"{cnf_file.stem}.csv"
            satzilla_logger.info(f"Computing features for: {cnf_file}")

            cmd = [satzilla_path, "-base", str(cnf_file), str(output_file)]
            return_code, _, stderr = run_command(cmd)

            if return_code != 0:
                satzilla_logger.warning(f"Error computing features for {cnf_file}: {stderr}")
                continue

            satzilla_logger.info(f"Features computed for: {cnf_file}")

        try:
            merged_df: pd.DataFrame = merge_feature_csvs(temp_path)
            enhanced_df: pd.DataFrame = extract_generator_seed_and_add_columns(merged_df)

            output_file: Path = features_output_path / output_file_name
            enhanced_df.to_csv(output_file, index=False)
            satzilla_logger.info(f"Merged features saved to: {output_file}")

        except ValueError as e:
            satzilla_logger.error(f"Error merging CSV files: {e}")

def process_csv_files(features_output_dir: str):
    """Process CSV files to remove duplicates."""
    for csv_file in os.listdir(features_output_dir):
        if csv_file.endswith(".csv"):
            csv_file_path = os.path.join(features_output_dir, csv_file)
            try:
                satzilla_logger.info(f"Processing CSV file: {csv_file_path}")
                df = pd.read_csv(csv_file_path, header=None)
                first_row = df.iloc[0]
                df = df[df.ne(first_row).any(axis=1)]
                df.iloc[0] = first_row
                df.to_csv(csv_file_path, index=False, header=False)
                satzilla_logger.info(f"Processed successfully: {csv_file_path}")
            except Exception as e:
                satzilla_logger.error(f"Error processing {csv_file_path}: {e}")

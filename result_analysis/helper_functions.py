import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, Any, List

def process_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path)
    if "instance_path" in data.columns:
        data["instance_name"] = data["instance_path"].apply(
            lambda x: os.path.basename(x).replace(".cnf", "")
        )
        data.drop(columns=["instance_path"], inplace=True)
    drop_cols = [
        "est_val", "instance_path", "Unnamed: 0",
        "est_type", "counter_type", "count_precision", "count_notation", "instance"
    ]
    data_cleaned = data.drop(columns=[col for col in drop_cols if col in data.columns], errors="ignore")
    if "instance_name" in data_cleaned.columns:
        data_cleaned = data_cleaned.set_index("instance_name")
    return data_cleaned

def add_seed_magnitude_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column for seed magnitude to the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'seed' column.

    Returns:
        pd.DataFrame: DataFrame with added 'seed_magnitude' column.
    """
    data["seed_magnitude"] = data["seed"].apply(
        lambda x: len(str(int(abs(x)))) if pd.notnull(x) else None
    )
    return data

def add_generator_iter_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'generator_iter_number' column to the DataFrame by extracting iteration numbers
    from the 'instance_name' column or index. Assumes the iteration number is in the format
    '_<number>_s' within the 'instance_name'.

    Args:
        data (pd.DataFrame): DataFrame containing 'instance_name' in its columns or index.

    Returns:
        pd.DataFrame: Updated DataFrame with an added 'generator_iter_number' column, where:
                      - Numbers are extracted from 'instance_name'.
                      - Missing or unmatched values are replaced with -1.
    """
    instance_source: pd.Series = (
        data["instance_name"] if "instance_name" in data.columns else data.index.to_series()
    )
    data["generator_iter_number"] = (
        instance_source
        .str.extract(r"_(\d+)_s")[0]
        .fillna(-1)
        .astype(int)
        .values
    )
    return data

def add_seed_index_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a seed index column to the DataFrame based on the length of seed magnitudes.

    Args:
        data (pd.DataFrame): DataFrame containing a 'seed' column.

    Returns:
        pd.DataFrame: DataFrame with added 'seed_index' column.
    """
    data["seed_index"] = np.where(
        data["seed"].notnull() & data["seed"].abs().astype(int).astype(str).str.len().isin([1, 2]),
        0,
        np.where(
            data["seed"].notnull() & data["seed"].abs().astype(int).astype(str).str.len() == 3,
            1,
            data["seed"].abs().astype(int).astype(str).str.len()
        )
    )
    return data

def process_generator_column(data: pd.DataFrame, ordered_generators: List[str]) -> pd.DataFrame:
    """
    Process the 'generator' column in the DataFrame to be categorical and add an encoded version.

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'generator' column.
        ordered_generators (List[str]): Ordered list of generator categories.

    Returns:
        pd.DataFrame: DataFrame with 'generator' column processed and encoded.
    """
    if "generator" not in data.columns:
        raise ValueError("'generator' column is missing from the DataFrame.")

    data["generator"] = pd.Categorical(
        data["generator"],
        categories=ordered_generators,
        ordered=True
    )
    data["generator_encoded"] = data["generator"].cat.codes
    return data

def extract_numeric_features(data: pd.DataFrame) -> List[str]:
    """
    Extract the names of numeric features (float64 and int64) from the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        List[str]: List of column names with numeric data types.
    """
    return data.select_dtypes(include=["float64", "int64"]).columns.tolist()

def post_process_general_instances(data: pd.DataFrame) -> List[str]:
    """
    Post-process general instances by encoding the 'generator' column and extracting numeric features.

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'generator' column.

    Returns:
        List[str]: List of numeric feature column names.
    """
    ordered_generators = [
        "FuzzSAT-easy",
        "FuzzSAT-medium",
        "FuzzSAT-hard",
        "FuzzSAT-mixed",
        "FuzzSAT-random-medium",
        "FuzzSAT-random-hard",
        "FuzzSAT-structured-hard"
    ]

    data = process_generator_column(data, ordered_generators)
    numeric_features = extract_numeric_features(data)

    return numeric_features

def analyze_learning_curves(
    data: pd.DataFrame,
    numeric_features: List[str],
    target_column: str
) -> pd.DataFrame:
    """
    Analyze learning curves by iteratively removing samples and calculating accuracy.

    Args:
        data (pd.DataFrame): DataFrame containing features and target column.
        numeric_features (List[str]): List of numeric feature column names.
        target_column (str): Name of the target column.

    Returns:
        pd.DataFrame: DataFrame containing iteration results with remaining samples, accuracy, and error rate.
    """
    unique_seed_indices = np.sort(data["seed_index"].unique())
    unique_iter_numbers = np.sort(data["generator_iter_number"].unique())

    results = []
    current_data = data.copy()

    # Initialize classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )

    for seed_index in unique_seed_indices:
        for iter_number in unique_iter_numbers:
            # Remove samples in batches
            mask = (current_data["seed_index"] != seed_index) | (
                current_data["generator_iter_number"] != iter_number
            )
            current_data = current_data[mask]
            remaining_samples = len(current_data)

            if remaining_samples > 10:
                # Prepare data for stratified cross-validation
                X_current = current_data[numeric_features].values
                y_current = current_data[target_column].values

                # Adjust the number of splits dynamically
                min_class_size = current_data[target_column].value_counts().min()
                adjusted_splits = min(3, min_class_size)

                if adjusted_splits > 1:
                    skf = StratifiedKFold(n_splits=adjusted_splits, shuffle=True, random_state=42)
                    cv_scores = [
                        rf_classifier.fit(X_current[train_idx], y_current[train_idx])
                        .score(X_current[test_idx], y_current[test_idx])
                        for train_idx, test_idx in skf.split(X_current, y_current)
                    ]
                    accuracy = np.mean(cv_scores)
                    error_rate = 1 - accuracy
                else:
                    accuracy = None
                    error_rate = None
            else:
                accuracy = None
                error_rate = None

            # Store results
            results.append({
                "remaining_samples": remaining_samples,
                "accuracy": accuracy,
                "error_rate": error_rate,
            })

    return pd.DataFrame(results).dropna()

def merge_data(
    fuzzing_results: pd.DataFrame,
    satzilla_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge invalid fuzzer instances with Satzilla features, based on their index
    (which should be set to 'instance_name' in both data frames).

    Args:
        fuzzing_results (pd.DataFrame): DataFrame of invalid fuzzer instances.
        satzilla_features (pd.DataFrame): DataFrame of Satzilla features.

    Returns:
        pd.DataFrame: Merged DataFrame, containing rows that appear in both.
    """
    # Make sure both DataFrames have 'instance_name' set as index
    if "instance_name" in fuzzing_results.columns:
        fuzzing_results = fuzzing_results.set_index("instance_name")
    if "instance_name" in satzilla_features.columns:
        satzilla_features = satzilla_features.set_index("instance_name")

    satzilla_features.drop(columns=["generator"], inplace=True, errors="ignore")
    # Merge on the index
    merged_df = fuzzing_results.join(satzilla_features, how="inner")
    return merged_df

def classify_fuzzing_results(merged_df: pd.DataFrame) -> List[pd.DataFrame]:
   """
   Classify fuzzing results into categories:
   1. TIMEOUT: Where timed_out is True
   2. CRASH: Where satisfiability or count_value is NaN/Null
   3. CORRECT COUNT: Where count_matches is True
   4. INCORRECT COUNT: Where count_matches is False and cpog_message is "NO ERROR",
      or where count_values don't match the majority within an instance group
   5. CPOG errors: Remaining instances with consistent count_values, grouped by cpog_message

   Args:
       merged_df (pd.DataFrame): DataFrame (already merged with Satzilla features)
                               containing fuzzing results and relevant columns.

   Returns:
       List[pd.DataFrame]: A list containing DataFrames for each category:
           - timeout_results
           - crash_results
           - correct_results
           - incorrect_results
           followed by DataFrames for each CPOG error message
   """
   # 1. TIMEOUT
   timeout_mask = merged_df["timed_out"] == True if "timed_out" in merged_df.columns else pd.Series(False, index=merged_df.index)
   timeout_results = merged_df[timeout_mask].copy()
   remaining_df = merged_df[~timeout_mask].copy()

   # 2. CRASH
   crash_mask = remaining_df["satisfiability"].isna() | remaining_df["count_value"].isna()
   crash_results = remaining_df[crash_mask].copy()
   remaining_df = remaining_df[~crash_mask].copy()

   # 3. CORRECT COUNT
   correct_mask = remaining_df["count_matches"] == True
   correct_results = remaining_df[correct_mask].copy()
   remaining_df = remaining_df[~correct_mask].copy()

   # 4. INCORRECT COUNT - first pass for clear incorrect counts
   incorrect_mask = (remaining_df["count_matches"] == False) & (remaining_df["cpog_message"] == "NO ERROR")
   incorrect_results = remaining_df[incorrect_mask].copy()
   remaining_df = remaining_df[~incorrect_mask].copy()

   # 5. Process remaining instances - check count value consistency
   cpog_errors_list = []
   if not remaining_df.empty:
       # Group by instance_name and check count_value consistency
       instance_groups = remaining_df.groupby(level=0)
       
       additional_incorrect = pd.DataFrame()
       consistent_cpog_errors = pd.DataFrame()
       
       for _, group in instance_groups:
           unique_counts = group["count_value"].unique()
           if len(unique_counts) > 1:
               # Multiple count values - do majority vote
               count_value_counts = group["count_value"].value_counts()
               majority_count = count_value_counts.index[0]
               incorrect_mask = group["count_value"] != majority_count
               additional_incorrect = pd.concat([additional_incorrect, group[incorrect_mask]])
           else:
               # Consistent count values - this is a CPOG error
               consistent_cpog_errors = pd.concat([consistent_cpog_errors, group])
       
       # Add instances with inconsistent counts to incorrect_results
       incorrect_results = pd.concat([incorrect_results, additional_incorrect])
       
       # Group remaining CPOG errors by message
       if not consistent_cpog_errors.empty:
           for msg in consistent_cpog_errors["cpog_message"].unique():
               if pd.notna(msg):
                   subset = consistent_cpog_errors[consistent_cpog_errors["cpog_message"] == msg].copy()
                   cpog_errors_list.append(subset)

   return [timeout_results, crash_results, correct_results, incorrect_results] + cpog_errors_list

def perform_stratified_rf_analysis(
    data: pd.DataFrame,
    numeric_features: List[str],
    target_column: str,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, pd.DataFrame]:
    """
    Perform stratified random forest analysis to identify important features.

    Args:
        data (pd.DataFrame): DataFrame containing features and target.
        numeric_features (List[str]): List of numeric feature names.
        target_column (str): Name of the target column.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[RandomForestClassifier, pd.DataFrame]: Trained model and feature importances.
    """
    X = data[numeric_features].values
    y = data[target_column].values

    rf_classifier = RandomForestClassifier(
        n_estimators=100, random_state=random_state, class_weight="balanced"
    )
    rf_classifier.fit(X, y)

    feature_importances = pd.DataFrame({
        "Feature": numeric_features,
        "Importance": rf_classifier.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return rf_classifier, feature_importances

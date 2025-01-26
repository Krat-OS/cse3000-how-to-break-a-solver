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
    Load a CSV file into a pandas DataFrame, handling multiple CSV formats.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Processed DataFrame with consistent formatting.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = pd.read_csv(file_path)
    
    # Drop 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)
    
    if "instance" in data.columns:
        # Create instance_name from instance_path
        data["instance_name"] = data["instance"].apply(
            lambda x: os.path.basename(x).replace(".cnf", "")
        )
        data.drop(columns=["instance"], inplace=True)
    elif "instance_name" not in data.columns:
        # If neither instance nor instance_name exists,
        # try to create it from generator and seed
        if "generator" in data.columns and "seed" in data.columns:
            data["instance_name"] = data.apply(
                lambda row: f"{row['generator']}_{row['seed']}",
                axis=1
            )
        else:
            raise ValueError("Could not find 'instance' or 'instance_name' columns.")
    
    # Transform count_value to float instead of integer to handle NaN properly
    if 'count_value' in data.columns:
        data['count_value'] = pd.to_numeric(data['count_value'], errors='coerce')
    
    # Extract generator based on the naming pattern
    def extract_generator(x):
        parts = x.split('_')[0].split('-')  # Split before underscore, then by hyphen
        if len(parts) >= 3:  # If we have at least 3 parts (generator, difficulty, randomness)
            return f"{parts[0]}-{parts[1]}-{parts[2]}"
        return x  # Return original if pattern doesn't match
    
    data["generator"] = data["instance_name"].apply(extract_generator)
    
    # Extract base generator
    data["base_generator"] = data["instance_name"].apply(
        lambda x: x.split("-")[0]
    )
    
    # Extract presumed difficulty
    data["presumed_difficulty"] = data["instance_name"].apply(
        lambda x: x.split("-")[1]
    )
    
    # Extract randomness value
    data["randomness"] = data["instance_name"].apply(
        lambda x: int(x.split("-")[2].split("_")[0])
    )
    
    # Always set index to instance_name
    if "instance_name" in data.columns:
        data = data.set_index("instance_name")
    
    return data

def get_numeric_features(df):
    """Get numeric features, excluding metadata and time columns"""
    features_to_exclude = ['seed', 'randomness', 'solved', 'Pre-featuretime', 'Basic-featuretime', 'KLB-featuretime', 'CG-featuretime', 'solve_time', "count_value", "est_val"]
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    return [col for col in numeric_cols if col not in features_to_exclude]

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

def _classify_with_cpog(merged_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Classify fuzzing results using CPOG verification.
    """
    # 3. CORRECT COUNT
    correct_mask = merged_df["count_matches"] == True
    correct_results = merged_df[correct_mask].copy()
    remaining_df = merged_df[~correct_mask].copy()
    
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
    
    return [correct_results, incorrect_results] + cpog_errors_list

def _classify_with_majority_vote(merged_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Classify fuzzing results using majority vote across counters.
    """
    # Add majority vote column
    instance_groups = merged_df.groupby(level=0)
    majority_votes = {}
    
    for instance_name, group in instance_groups:
        count_value_counts = group["count_value"].value_counts()
        majority_count = count_value_counts.index[0]
        majority_votes[instance_name] = majority_count
    
    merged_df["correct_count_majority_vote"] = merged_df.index.map(majority_votes)
    merged_df["count_matches_majority"] = merged_df["count_value"] == merged_df["correct_count_majority_vote"]
    
    # Classify based on majority vote
    correct_mask = merged_df["count_matches_majority"] == True
    correct_results = merged_df[correct_mask].copy()
    incorrect_results = merged_df[~correct_mask].copy()
    
    return [correct_results, incorrect_results]

def classify_fuzzing_results(merged_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Classify fuzzing results into categories, using either CPOG verification or majority voting
    depending on available columns.
    
    Args:
        merged_df (pd.DataFrame): DataFrame containing fuzzing results
        
    Returns:
        List[pd.DataFrame]: A list containing DataFrames for each category:
        - timeout_results
        - crash_results
        - correct_results
        - incorrect_results
        followed by DataFrames for each CPOG error message (if using CPOG verification)
    """
    # 1. TIMEOUT
    timeout_mask = merged_df["timed_out"] == True if "timed_out" in merged_df.columns else pd.Series(False, index=merged_df.index)
    timeout_results = merged_df[timeout_mask].copy()
    remaining_df = merged_df[~timeout_mask].copy()
    
    # 2. CRASH
    crash_mask = remaining_df["satisfiability"].isna() & remaining_df["count_value"].isna()
    crash_results = remaining_df[crash_mask].copy()
    remaining_df = remaining_df[~crash_mask].copy()
    
    # Check if CPOG columns exist
    has_cpog = all(col in remaining_df.columns for col in ["cpog_message", "cpog_count", "count_matches"])
    
    # Use appropriate classification method
    if has_cpog:
        classification_results = _classify_with_cpog(remaining_df)
    else:
        classification_results = _classify_with_majority_vote(remaining_df)
    
    return [timeout_results, crash_results] + classification_results

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

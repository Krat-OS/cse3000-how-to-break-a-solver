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
    "est_val", "count_value", "instance_path", "Unnamed: 0",
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
  data['seed_magnitude'] = data['seed'].apply(
    lambda x: len(str(int(abs(x)))) if pd.notnull(x) else None
  )
  return data

def add_generator_iter_column(data: pd.DataFrame) -> pd.DataFrame:
  """
  Extract generator iteration numbers from 'instance_name' and add as a column.

  Args:
      data (pd.DataFrame): DataFrame containing 'instance_name' column.

  Returns:
      pd.DataFrame: DataFrame with added 'generator_iter_number' column.
  """
  data['generator_iter_number'] = (
    data.index
    .str.extract(r"_(\d+)_s")[0]  # Extract matched group or None
    .fillna(-1)  # Replace NaN with a placeholder (-1)
    .astype(int)  # Convert to integers
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
  data['seed_index'] = np.where(
    data['seed'].notnull() & data['seed'].abs().astype(int).astype(str).str.len().isin([1, 2]),
    0,
    np.where(
      data['seed'].notnull() & data['seed'].abs().astype(int).astype(str).str.len() == 3,
      1,
      data['seed'].abs().astype(int).astype(str).str.len()
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
  if 'generator' not in data.columns:
    raise ValueError("'generator' column is missing from the DataFrame.")

  data['generator'] = pd.Categorical(
    data['generator'],
    categories=ordered_generators,
    ordered=True
  )
  data['generator_encoded'] = data['generator'].cat.codes
  return data

def extract_numeric_features(data: pd.DataFrame) -> List[str]:
  """
  Extract the names of numeric features (float64 and int64) from the DataFrame.

  Args:
      data (pd.DataFrame): Input DataFrame.

  Returns:
      List[str]: List of column names with numeric data types.
  """
  return data.select_dtypes(include=['float64', 'int64']).columns.tolist()

def post_process_general_instances(data: pd.DataFrame) -> List[str]:
  """
  Post-process general instances by encoding the 'generator' column and extracting numeric features.

  Args:
      data (pd.DataFrame): Input DataFrame containing a 'generator' column.

  Returns:
      List[str]: List of numeric feature column names.
  """
  ordered_generators = [
    'brummayer-structured-medium',
    'brummayer-medium',
    'brummayer-random-med',
    'brummayer-structured-hard',
    'brummayer-hard',
    'brummayer-random-hard',
    'brummayer-mixed'
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
  unique_seed_indices = np.sort(data['seed_index'].unique())
  unique_iter_numbers = np.sort(data['generator_iter_number'].unique())

  results = []
  current_data = data.copy()

  # Initialize classifier
  rf_classifier = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced'
  )

  for seed_index in unique_seed_indices:
    for iter_number in unique_iter_numbers:
      # Remove samples in batches
      mask = (current_data['seed_index'] != seed_index) | (
        current_data['generator_iter_number'] != iter_number
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
        'remaining_samples': remaining_samples,
        'accuracy': accuracy,
        'error_rate': error_rate,
      })

  return pd.DataFrame(results).dropna()

def plot_learning_curves(
  results_df: pd.DataFrame,
  accuracy_threshold: float = 0.95,
  error_rate_threshold: float = 0.05,
  min_seeds: int = 4,
  min_iters_per_gen: int = 30,
  cmap_name: str = "viridis"
):
  """
  Plot performance metrics (accuracy and error rate) versus remaining samples.

  Args:
      results_df (pd.DataFrame): DataFrame containing learning curve results.
      accuracy_threshold (float): Accuracy threshold for highlighting results.
      error_rate_threshold (float): Error rate threshold for highlighting results.
      min_seeds (int): Minimum number of seeds for annotations.
      min_iters_per_gen (int): Minimum iterations per generator for annotations.
      cmap_name (str): Name of the Matplotlib colormap to use (e.g., 'viridis').

  Returns:
      None
  """
  # Get the colormap
  colormap = cm.get_cmap(cmap_name)

  plt.figure(figsize=(12, 6))

  # Plot accuracy, using a color from the selected colormap
  plt.plot(
    results_df['remaining_samples'],
    results_df['accuracy'],
    label='Accuracy',
    marker='o',
    markersize=4,
    linestyle='-',
    linewidth=1.5,
    color=colormap(0.3),  # pick any fraction between 0 and 1
    alpha=0.7,
  )

  # Plot error rate, using a different color from the same colormap
  plt.plot(
    results_df['remaining_samples'],
    results_df['error_rate'],
    label='Error Rate',
    marker='o',
    markersize=4,
    linestyle='-',
    linewidth=1.5,
    color=colormap(0.7),
    alpha=0.7,
  )

  # Highlight thresholds
  plt.axhline(
    y=accuracy_threshold,
    color='green',
    linestyle='--',
    linewidth=1.2,
    label='Accuracy Threshold',
  )
  plt.axhline(
    y=error_rate_threshold,
    color='red',
    linestyle='--',
    linewidth=1.2,
    label='Error Rate Threshold',
  )

  # Add annotations for seeds and iterations
  plt.text(
    x=100,  # adjust as needed
    y=0.2,  # adjust as needed
    s=f"Min # seeds: {min_seeds}\nMin # iter/gen: {min_iters_per_gen}",
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'),
  )

  # Set labels, title, and legend
  plt.xlabel('Remaining Samples', fontsize=12)
  plt.ylabel('Performance Metrics', fontsize=12)
  plt.title('Performance Metrics vs. Remaining Samples (Iterative Removal)', fontsize=14)
  plt.legend(loc='upper right', fontsize=10)
  plt.grid(visible=True, linestyle='--', alpha=0.6)

  # Optimize layout
  plt.tight_layout()
  plt.show()


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

def separate_fuzzing_results(merged_df: pd.DataFrame) -> List[pd.DataFrame]:
  """
  Separate the merged invalid fuzzer instances into:

  1. Correct results (where count_matches == True)
  2. Incorrect results (where count_matches == False and cpog_message == "NO ERROR")
  3. Fuzzer timeout errors (where timed_out == True)
  4. CPOG errors, grouped by cpog_message (everything remaining after steps 1-3).

  Args:
      merged_df (pd.DataFrame): DataFrame (already merged with Satzilla features)
                                containing invalid results and relevant columns.

  Returns:
      List[pd.DataFrame]: A list whose first elements are:
          - correct_results
          - incorrect_results
          - fuzzer_timeout
        followed by one DataFrame per unique CPOG message among the remaining rows.
  """
  # 1) Correct results: count_matches == True
  correct_results = merged_df[merged_df["count_matches"] == True].copy()
  merged_df.drop(index=correct_results.index, inplace=True)

  # 2) Incorrect results: count_matches == False AND cpog_message == "NO ERROR"
  incorrect_results = merged_df[
    (merged_df["count_matches"] == False) &
    (merged_df["cpog_message"] == "NO ERROR")
    ].copy()
  merged_df.drop(index=incorrect_results.index, inplace=True)

  # 3) Fuzzer timeouts: timed_out == True
  #    (If 'timed_out' does not exist, this will produce an empty DataFrame)
  if "timed_out" in merged_df.columns:
    fuzzer_timeout = merged_df[merged_df["timed_out"] == True].copy()
    merged_df.drop(index=fuzzer_timeout.index, inplace=True)
  else:
    fuzzer_timeout = pd.DataFrame()

  # 4) Remaining rows => split into CPOG error subsets by cpog_message
  cpog_errors_list = []
  if not merged_df.empty and "cpog_message" in merged_df.columns:
    for msg in merged_df["cpog_message"].unique():
      subset = merged_df[merged_df["cpog_message"] == msg].copy()
      cpog_errors_list.append(subset)

  # Return the results
  return [correct_results, incorrect_results, fuzzer_timeout] + cpog_errors_list

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
    n_estimators=100, random_state=random_state, class_weight='balanced'
  )
  rf_classifier.fit(X, y)

  feature_importances = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': rf_classifier.feature_importances_
  }).sort_values(by='Importance', ascending=False)

  return rf_classifier, feature_importances


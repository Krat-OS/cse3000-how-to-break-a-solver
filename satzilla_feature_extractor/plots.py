import pandas as pd
import matplotlib.pyplot as plt

def get_features_list(csv_file_path):
    """
    Get a list of all features (columns) in the CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        list: List of feature names (column headers).
    """
    df = pd.read_csv(csv_file_path)
    return list(df.columns)

def sort_csv_by_feature(csv_file_path, feature, output_file_path=None):
    """
    Sorts a CSV file by a specific feature in increasing order and saves the result.

    Args:
        csv_file_path (str): Path to the input CSV file.
        feature (str): Name of the feature (column) to sort by.
        output_file_path (str, optional): Path to save the sorted CSV file. If None, overwrites the input file.

    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Check if the feature exists in the DataFrame
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in the CSV file.")

    # Sort the DataFrame by the specified feature
    sorted_df = df.sort_values(by=feature, ascending=True)

    # Save the sorted DataFrame to the specified output file
    if output_file_path is None:
        output_file_path = csv_file_path  # Overwrite the input file

    sorted_df.to_csv(output_file_path, index=False)

    print(f"CSV file sorted by '{feature}' and saved to: {output_file_path}")

    return sorted_df

def sort_csv_by_instance_name(csv_file_path):
    """
    Sort a CSV file by the numeric part of the `instance` column and save it back to the file.

    Args:
        csv_file_path (str): Path to the CSV file.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the numeric part from the `instance` column and create a sorting key
    df['instance_number'] = df['instance'].str.extract(r'(\d+)(?=\.cnf$)').astype(int)

    # Sort the DataFrame by the extracted instance number
    df_sorted = df.sort_values(by='instance_number')

    # Drop the temporary sorting column
    df_sorted = df_sorted.drop(columns=['instance_number'])

    # Save the sorted DataFrame back to the same file
    df_sorted.to_csv(csv_file_path, index=False)

    print(f"Sorted CSV file saved to {csv_file_path}")

def plot_solve_time(csv_file_path, a=None, b=None):
    """
    Plot the solve time for each instance in the CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.
        a (int, optional): Start index of the range to plot. Default is None (start at 0).
        b (int, optional): End index of the range to plot. Default is None (plot until the end).
    """
    sort_csv_by_instance_name(csv_file_path)
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Check if the 'solve_time' column exists
    if 'solve_time' not in df.columns:
        raise ValueError("The CSV file does not contain a 'solve_time' column.")

    # Extract the solve time values
    solve_times = pd.to_numeric(df['solve_time'], errors='coerce').dropna()

    # Apply index range if specified
    if a is not None and b is not None:
        if a < 0 or b > len(solve_times):
            raise ValueError(f"Indices out of range: a={a}, b={b}, length={len(solve_times)}")
        solve_times = solve_times.iloc[a:b]
        indices = range(a, b)  # Adjust x-axis for subset
    else:
        indices = range(len(solve_times))

    # Plot the solve times with connected dots
    plt.figure(figsize=(12, 6))
    plt.plot(indices, solve_times, color='blue', alpha=0.7, marker='o', label='Solve Time')
    plt.title("Solve Time for Each Instance", fontsize=14)
    plt.xlabel("Instance Index", fontsize=12)
    plt.ylabel("Solve Time (seconds)", fontsize=12)
    plt.legend()
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



def plot_feature_values_combined(csv_files, feature, range_min=None, range_max=None, labels=None):
    """
    Plot values for a single feature across multiple CSV files on a single plot.

    Args:
        csv_files (list): List of paths to CSV files.
        feature (str): Name of the feature to plot.
        range_min (float, optional): Minimum value for the y-axis. Default is None.
        range_max (float, optional): Maximum value for the y-axis. Default is None.
        labels (list, optional): List of labels for each CSV file. Default is None.
    """
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(csv_files))]
    elif len(labels) != len(csv_files):
        raise ValueError("Length of labels must match the number of CSV files.")

    colors = plt.cm.tab10.colors  # Use a set of distinct colors
    plt.figure(figsize=(12, 6))

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        values = df[feature]
        plt.plot(range(len(values)), values, label=labels[i], color=colors[i % len(colors)], marker='o', alpha=0.7)

    # Set y-axis range if range_min and range_max are provided
    if range_min is not None and range_max is not None:
        plt.ylim(range_min, range_max)

    # Add title, labels, legend, and grid
    # plt.title(f"Combined Plot for Feature: {feature}", fontsize=20)
    plt.xlabel("Instance Index", fontsize=18)
    plt.ylabel(feature, fontsize=18)
    plt.tick_params(axis='both', labelsize=17)  # Increase font size for axis ticks
    plt.legend(fontsize=17)     
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_feature_values(csv_file_path, feature, range_min=None, range_max=None):
    """
    Plot all values for a single feature across all instances as dots,
    with a minimal grid. Optionally, set the y-axis range.

    Args:
        csv_file_path (str): Path to the CSV file.
        feature (str): Name of the feature to plot.
        range_min (float, optional): Minimum value for the y-axis. Default is None.
        range_max (float, optional): Maximum value for the y-axis. Default is None.
    """
    df = pd.read_csv(csv_file_path)
    values = df[feature]

    plt.figure(figsize=(10, 6))
    # Plot only dots
    plt.scatter(range(len(values)), values, color='blue', s=50, alpha=0.7)

    # Add a grid with fewer lines
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Fewer X ticks
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(10))  # Fewer Y ticks

    # Set y-axis range if range_min and range_max are provided
    if range_min is not None and range_max is not None:
        plt.ylim(range_min, range_max)

    # Add title and labels
    plt.title(f"Values for Feature: {feature}", fontsize=14)
    plt.xlabel("Instance Index", fontsize=12)
    plt.ylabel(feature, fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_all_feature_values(csv_file_path, features):
    """Plot all selected features on separate plots with different colors."""
    for feature in features:
        plot_feature_values(csv_file_path, feature)


def plot_feature_values_with_lines(csv_file_path, feature):
    """
    Plot all values for a single feature with vertical lines scaled to the y-axis range.

    Args:
        csv_file_path (str): Path to the CSV file.
        feature (str): Name of the feature to plot.
    """
    df = pd.read_csv(csv_file_path)

    # Convert feature values to numeric, forcing invalid entries to NaN
    values = pd.to_numeric(df[feature], errors='coerce').dropna()
    max_value = values.max()  # Get the maximum value for normalization

    if max_value == 0 or max_value is None:
        raise ValueError(f"Feature '{feature}' has no valid numeric values to plot.")

    plt.figure(figsize=(12, 6))

    # Normalize feature values and plot vertical lines
    for i, value in enumerate(values[:200]):  # Limit to 200 instances for clarity
        normalized_value = value / max_value
        plt.axvline(x=i, ymin=0, ymax=normalized_value, color='blue', alpha=0.6)

    # Add dots for clarity
    plt.scatter(range(len(values[:200])), values[:200], color='red', s=10, alpha=0.8, label='Feature Values')

    plt.title(f"Feature: {feature} (Vertical Line Plot)")
    plt.xlabel("Instance Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_2d_features(csv_file_path, feature_x, feature_y):
    """
    Plot a 2D scatter plot for two chosen features.
    
    Args:
        csv_file_path (str): Path to the CSV file.
        feature_x (str): Name of the feature for the X-axis.
        feature_y (str): Name of the feature for the Y-axis.
    """
    df = pd.read_csv(csv_file_path)
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature_x], df[feature_y], color='green', alpha=0.7)
    plt.title(f"Scatter Plot of {feature_x} vs {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.grid(True)
    plt.show()


def plot_instance_values(csv_file_path, index):
    """
    Plot all feature values for a single instance (row).
    
    Args:
        csv_file_path (str): Path to the CSV file.
        index (int): Index of the instance to plot.
    """
    df = pd.read_csv(csv_file_path)
    instance = df.iloc[index]
    features = instance.index
    values = instance.values

    plt.figure(figsize=(12, 6))
    plt.bar(features, values, color='orange')
    plt.title(f"Values for Instance at Index: {index}")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def compare_instances(csv_file_path, index1, index2):
    """
    Compare two instances side-by-side by plotting their feature values.
    
    Args:
        csv_file_path (str): Path to the CSV file.
        index1 (int): Index of the first instance to compare.
        index2 (int): Index of the second instance to compare.
    """
    df = pd.read_csv(csv_file_path)
    instance1 = df.iloc[index1]
    instance2 = df.iloc[index2]
    features = instance1.index
    values1 = instance1.values
    values2 = instance2.values

    x = range(len(features))
    plt.figure(figsize=(14, 6))
    plt.bar(x, values1, width=0.4, label=f"Instance {index1}", color='blue', align='center')
    plt.bar(x, values2, width=0.4, label=f"Instance {index2}", color='red', align='edge')
    plt.xticks(x, features, rotation=45, ha="right")
    plt.title(f"Comparison of Features for Instances {index1} and {index2}")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.legend()
    plt.tight_layout()
    plt.show()

def sort_csv_by_column(csv_file_path, column_name, output_file_path):
    """
    Sort a CSV file by a specific column and save the result to a new file.

    Args:
        csv_file_path (str): Path to the input CSV file.
        column_name (str): Name of the column to sort by.
        output_file_path (str): Path to save the sorted CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Check if the column exists in the CSV
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file.")

    # Sort the DataFrame by the specified column
    sorted_df = df.sort_values(by=column_name, ascending=True)

    # Save the sorted DataFrame to a new file
    sorted_df.to_csv(output_file_path, index=False)

    print(f"CSV file sorted by '{column_name}' and saved to {output_file_path}.")

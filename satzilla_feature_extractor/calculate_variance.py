import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

theoretical_ranges = {
    "cluster-coeff-mean": (0, 1),
    "VCG-VAR-mean": (0, 1),
    "VCG-CLAUSE-mean": (0, 1),
    "vars-clauses-ratio": (0, 1),
    "horn-clauses-fraction": (0, 1),
    "reducedClauses": (0, 500),
    "reducedVars": (0, 150),
    "BINARY+": (0, 1),
    "TRINARY+": (0, 1)
}

def get_delft_blue_shades():
    """Generate 10 shades of Delft Blue."""
    return ListedColormap([
        '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
        '#6baed6', '#4292c6', '#2171b5', '#08519c',
        '#08306b', '#041526'
    ])

def plot_variance_table(csv_file_path, k=1, normalize=False, fs=12):
    """
    Plot selected features in a 3x3 table with optional normalization.

    Args:
        csv_file_path (str): Path to the CSV file.
        theoretical_ranges (dict): Dictionary of theoretical min and max for each feature.
        k (float): Power factor for normalization contrast.
        normalize (bool): Whether to apply normalization with observed and theoretical ranges. Default is False.
        fs (int): Font size for the text inside the boxes. Default is 12.
    """
    selected_features = list(theoretical_ranges.keys())

    # Read CSV
    df = pd.read_csv(csv_file_path)

    # Filter only the selected features
    df = df[selected_features]

    # Step 1: Calculate Coefficient of Variation (CV) for each feature
    cv_values = {}
    for feature in selected_features:
        mean = df[feature].mean()
        std_dev = df[feature].std()

        # Avoid division by zero
        if mean == 0:
            cv_values[feature] = std_dev / 0.0001
        else:
            # Calculate CV
            cv_values[feature] = std_dev / mean

    if normalize:
        # Step 2: Adjust values based on observed and theoretical ranges
        adjusted_values = {}
        for feature in selected_features:
            obs_min = df[feature].min()
            obs_max = df[feature].max()
            theor_min, theor_max = theoretical_ranges[feature]

            # Ensure theoretical range is not zero
            if theor_max - theor_min == 0:
                raise ValueError(f"Theoretical range for {feature} cannot be zero.")

            # Adjust feature values
            adjustment_factor = (obs_max - obs_min) / (theor_max - theor_min)
            adjusted_values[feature] = adjustment_factor

        ## Normalize adjusted values to a range of 0-1
        min_adjusted = min(adjusted_values.values())
        max_adjusted = max(adjusted_values.values())
        intermediate_normalized_values = {
            feature: ((cv_values[feature] * adjusted_values[feature] - min_adjusted) /
                      (max_adjusted - min_adjusted)) ** k
            for feature in selected_features
        }

        # Final normalization to cap values at 1
        min_final = min(intermediate_normalized_values.values())
        max_final = max(intermediate_normalized_values.values())
        normalized_values = {
            # feature: (value - min_final) / (max_final - min_final)
            feature: value
            for feature, value in intermediate_normalized_values.items()
        }
    else:
        # Step 3: Normalize CV values directly
        min_cv = min(cv_values.values())
        max_cv = max(cv_values.values())
        normalized_values = {
            feature: ((cv - min_cv) / (max_cv - min_cv)) ** k
            for feature, cv in cv_values.items()
        }

    # Step 4: Plot the 3x3 grid
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_title("Feature Values (Normalized)", fontsize=16)
    colormap = get_delft_blue_shades()

    for idx, (feature, value) in enumerate(normalized_values.items()):
        row = idx // 3
        col = idx % 3

        color_idx = int(value * 9)
        color = colormap(color_idx)

        # Create the cell background
        rect = plt.Rectangle((col, row), 1, 1, color=color)
        ax.add_patch(rect)

        # Place the text in the center of the cell
        ax.text(
            col + 0.5, row + 0.5, f"{feature}\n{value:.5f}",
            ha='center', va='center', fontsize=fs,
            color='white' if value > 0.5 else 'black'
        )

    # Set limits for the 3x3 grid
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # Invert the y-axis so that row 0 is at the top
    ax.invert_yaxis()

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    # Print the normalized values (what's displayed in the heatmap)
    print("Normalized feature values used in the heatmap:")
    for feature, val in normalized_values.items():
        print(f"{feature}: {val:.6f}")

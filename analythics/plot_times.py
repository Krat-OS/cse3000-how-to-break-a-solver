import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors  # For enabling hover functionality
from collections import defaultdict

def plot_solve_times(seed):
    directory = "../SharpVelvet/out/"
    files = [f for f in os.listdir(directory) if seed in f and f.endswith(".csv")]

    if not files:
        print(f"No files found for seed: {seed}")
        sys.exit(1)

    plt.figure(figsize=(16, 8))  # Increased width and height for better visualization
    
    # Keep track of solver -> color
    solvers = {}
    
    # Keep track of solver -> total empty/None counts
    solver_empty_counts = defaultdict(int)
    
    # Keep track of solver -> whether error = True anywhere
    solver_errors = defaultdict(bool)

    # For hover annotations:
    all_scatter_points = []   # Each entry is a PathCollection for a solver's file(s)
    all_annotation_texts = [] # Parallel list storing the text for hover in each scatter
    
    # -- Loop over files --
    for file_name in files:
        file_path = os.path.join(directory, file_name)

        # Extract the solver name from the file name
        solver = file_name.split('_')[5].split('_fuzz-results')[0]
        print("Solver:", solver)

        # Read the CSV
        df = pd.read_csv(file_path)
        
        # 1) Extract instance short path (after "/instances/")
        df["instance_short"] = df["instance"].str.split("instances/").str[-1]

        # 2) Track empty/None for count_value
        if "count_value" in df.columns:
            df_empty_count = df["count_value"].isna().sum() + df["count_value"].eq("").sum()
            solver_empty_counts[solver] += df_empty_count

        # 3) Track error column
        if "error" in df.columns:
            if df["error"].any():
                solver_errors[solver] = True
        
        # Extract the two 3-digit numbers
        df['primary_num'] = df['instance'].str.extract(r'_(\d{3})_')[0].astype(int)
        df['secondary_num'] = df['instance'].str.extract(r'_(\d{3}).cnf$')[0].astype(int)
        # Calculate the index
        df['index'] = df['primary_num'] + df['secondary_num'] * 10

        # Plot the solver's data
        scatter = plt.scatter(
            df['index'],
            df['solve_time'],
            s=10,    # Size of the dots
            label=None
        )
        
        # Store solver color (use the first facecolor in the PathCollection)
        solvers[solver] = scatter.get_facecolor()[0]

        # Keep instance_short for hover text
        all_scatter_points.append(scatter)
        all_annotation_texts.append(df['instance_short'].tolist())

    # ---------------------------------------------------
    # Configure the plot
    # ---------------------------------------------------
    plt.xlabel("Instance Index")
    plt.ylabel("Solve Time (seconds)")
    plt.title(f"Solve Times for Seed {seed}")
    plt.xticks([0, 200, 400, 600, 800, 1000])
    plt.grid(True)

    # ---------------------------------------------------
    # Build a custom legend
    # ---------------------------------------------------
    legend_labels = []
    for solver, color in solvers.items():
        label_str = solver
        
        # Append ERROR if needed
        if solver_errors[solver]:
            label_str += " (ERROR)"
        
        # Append empty/None counts if > 0
        empty_count = solver_empty_counts[solver]
        if empty_count > 0:
            label_str += f" (empty/None: {empty_count})"
        
        # Create a legend handle
        legend_labels.append(
            plt.Line2D([0], [0],
                       marker='o',
                       color=color,
                       lw=0,
                       label=label_str)
        )
    plt.legend(handles=legend_labels, title="Solvers", loc="upper right")

    # ---------------------------------------------------
    # Add hover functionality via mplcursors
    # ---------------------------------------------------
    # Use multiple=True so we can attach to multiple scatter objects
    cursor = mplcursors.cursor(all_scatter_points, hover=True, multiple=True)

    @cursor.connect("add")
    def on_add(sel):
        # Identify which PathCollection is being hovered
        scatter_obj = sel.artist
        # Find the scatter's index in our list
        scatter_index = all_scatter_points.index(scatter_obj)
        # sel.index is the index of the hovered point in that PathCollection
        local_index = sel.index
        # Retrieve the short instance path
        instance_text = all_annotation_texts[scatter_index][local_index]
        sel.annotation.set_text(instance_text)
        # Optional: make annotation more visible
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    # ---------------------------------------------------
    # Save and show the plot
    # ---------------------------------------------------
    output_path = os.path.join(directory, f"solve_times_{seed}_dots_with_hover.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

    # Finally display the interactive window
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_times.py <seed>")
        sys.exit(1)

    seed = sys.argv[1]
    plot_solve_times(seed)

if __name__ == "__main__":
    main()
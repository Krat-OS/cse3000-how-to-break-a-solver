import sys
import os
import argparse
from decimal import Decimal
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors  # For enabling hover functionality
from collections import defaultdict

def plot_solve_times(seed, solver_filter=None,
                     xmin=None, xmax=None,
                     ymin=None, ymax=None,
                     y2min=None, y2max=None):
    """
    If solver_filter is None or 'all', plot all solvers side-by-side.
      - Then, for each instance, if multiple solvers have different 'count_value' for that instance,
        print a mismatch warning to the terminal.

    If solver_filter is a specific solver (e.g., "d4g"), plot only that solver.
      - Also plot its model counts on a separate y-axis (right side) as a line.

    Axis limits:
      - If xmin, xmax, ymin, ymax, y2min, y2max are provided, we set them.
      - Otherwise, Matplotlib auto-scales.
    """
    directory = "../SharpVelvet/out/"
    files = [f for f in os.listdir(directory) if seed in f and f.endswith(".csv")]

    plt.rcParams.update({'font.size': 16})  # Global font size

    if not files:
        print(f"No files found for seed: {seed}")
        sys.exit(1)

    # ---------------------------------------------------
    # Prepare a figure
    # ---------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(16, 8))  # Main axes for solve times

    # Keep track of:
    #  1) solver -> color
    #  2) solver -> total empty/None counts
    #  3) solver -> whether error = True anywhere
    solvers = {}
    solver_empty_counts = defaultdict(int)
    solver_errors = defaultdict(bool)

    # We'll store data for mismatch checks or second y-axis usage
    solver_dfs = defaultdict(list)

    # For hover annotations (solve time):
    all_solved_points = []
    all_unsolved_points = []
    all_solved_annotations = []
    all_unsolved_annotations = []

    # For mismatch checking across multiple solvers
    instance_counts = defaultdict(dict)

    # ---------------------------------------------------
    # Process each CSV file
    # ---------------------------------------------------
    for file_name in files:
        file_path = os.path.join(directory, file_name)

        # Example file name: "2025-01-12_chevu_000_s1779779729113289578_000_d4g_fuzz-results.csv"
        # Parse out the solver from index 5
        solver = file_name.split('_')[5].split('_fuzz-results')[0]

        # Filter out solvers if requested
        if solver_filter and solver_filter != "all":
            if solver != solver_filter:
                continue

        # Read CSV
        df = pd.read_csv(file_path)

        # Extract instance short path
        df["instance_short"] = df["instance"].str.split("instances/").str[-1]

        # Track error columns & empty/None counts
        if "error" in df.columns and df["error"].any():
            solver_errors[solver] = True

        if "count_value" in df.columns:
            df_empty_count = df["count_value"].isna().sum() + df["count_value"].eq("").sum()
            solver_empty_counts[solver] += df_empty_count

        # Extract numeric index
        df['primary_num'] = df['instance'].str.extract(r'_(\d{3})_')[0].astype(int)
        df['secondary_num'] = df['instance'].str.extract(r'_(\d{3}).cnf$')[0].astype(int)
        df['index'] = (df['primary_num'] + df['secondary_num'] * 10) / 10.1

        # Convert count_value to numeric if possible
        if "count_value" in df.columns:
            df['count_value'] = pd.to_numeric(df['count_value'], errors='coerce')

        # Split into solved vs unsolved
        solved = df[df["count_value"].notna()]
        unsolved = df[df["count_value"].isna()]

        # Plot solved instances on ax1
        solved_scatter = ax1.scatter(
            solved['index'],
            solved['solve_time'],
            s=12,
            marker='o',  # circle
            label=None
        )
        solvers[solver] = solved_scatter.get_facecolor()[0]

        all_solved_points.append(solved_scatter)
        all_solved_annotations.append(solved['instance_short'].tolist())

        # Plot unsolved instances on ax1
        unsolved_scatter = ax1.scatter(
            unsolved['index'],
            unsolved['solve_time'],
            s=60,
            marker='x',  # circle with X
            color=solved_scatter.get_facecolor()[0],
            label=None
        )
        all_unsolved_points.append(unsolved_scatter)
        all_unsolved_annotations.append(unsolved['instance_short'].tolist())

        # Keep the entire DF for line plotting or mismatch checks
        solver_dfs[solver].append(df)

        # Collect data for mismatch checking (only if showing all)
        if (solver_filter is None) or (solver_filter == "all"):
            for idx, row in df.iterrows():
                inst = row["instance_short"]
                cval = row.get("count_value", None)
                if pd.notna(cval):
                    instance_counts[inst][solver] = str(int(float(cval)))

    # ---------------------------------------------------
    # If nothing was plotted, exit
    # ---------------------------------------------------
    if not all_solved_points and not all_unsolved_points:
        print(f"No data found for solver filter '{solver_filter}'. Exiting.")
        return

    # ---------------------------------------------------
    # Plot formatting for solve time
    # ---------------------------------------------------
    ax1.set_xlabel("Horn Clause Fraction", fontsize=16)
    ax1.set_ylabel("Solve Time (seconds)", fontsize=16)
    ax1.grid(True)

    # ---------------------------------------------------
    # Build a custom legend for the solvers
    # ---------------------------------------------------
    legend_labels = []
    for solver, color in solvers.items():
        label_str = solver
        if solver_errors[solver]:
            label_str += " (ERROR)"
        empty_count = solver_empty_counts[solver]
        if empty_count > 0:
            label_str += f" (Unsolved: {empty_count})"
        legend_labels.append(
            plt.Line2D([0], [0],
                       marker='o',
                       color=color,
                       lw=0,
                       label=label_str)
        )
    if legend_labels:
        ax1.legend(handles=legend_labels, title="Solvers", loc="upper center", fontsize=16)

    # ---------------------------------------------------
    # Mismatch check if 'all' solvers are plotted
    # ---------------------------------------------------
    if solver_filter is None or solver_filter == "all":
        for inst, sdict in instance_counts.items():
            unique_vals = set(sdict.values())
            if len(unique_vals) > 1:
                print(f"[Mismatch Warning] Instance '{inst}' has multiple count_values: {sdict}")

    # ---------------------------------------------------
    # If a single solver is specified, plot count_value on a second y-axis
    # ---------------------------------------------------
    ax2 = None
    if solver_filter and solver_filter != "all":
        combined_df = pd.concat(solver_dfs[solver_filter], ignore_index=True)
        combined_df = combined_df.dropna(subset=['count_value'])
        if not combined_df.empty:
            combined_df.sort_values(by='index', inplace=True)
            ax2 = ax1.twinx()
            ax2.set_ylabel("Model Count", fontsize=16)
            ax2.plot(
                combined_df['index'],
                combined_df['count_value'],
                color='black',
                linestyle='-',
                label="Model Count"
            )

    # ---------------------------------------------------
    # Add hover functionality for the scatter points
    # ---------------------------------------------------
    cursor = mplcursors.cursor(all_solved_points + all_unsolved_points, hover=True, multiple=True)

    @cursor.connect("add")
    def on_add(sel):
        scatter_obj = sel.artist
        if scatter_obj in all_solved_points:
            scatter_index = all_solved_points.index(scatter_obj)
            annotation_text = all_solved_annotations[scatter_index][sel.index]
        else:
            scatter_index = all_unsolved_points.index(scatter_obj)
            annotation_text = all_unsolved_annotations[scatter_index][sel.index]
        sel.annotation.set_text(annotation_text)
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    @cursor.connect("remove")
    def on_remove(sel):
        sel.annotation.set_visible(False)

    # ---------------------------------------------------
    # Optional Axis Limits
    # ---------------------------------------------------
    # x-axis
    if xmin is not None or xmax is not None:
        ax1.set_xlim(left=xmin, right=xmax)

    # primary y-axis (solve time)
    if ymin is not None or ymax is not None:
        ax1.set_ylim(bottom=ymin, top=ymax)

    # secondary y-axis (model count), if it exists
    if ax2 is not None:
        if y2min is not None or y2max is not None:
            ax2.set_ylim(bottom=y2min, top=y2max)

    # ---------------------------------------------------
    # Save and show the figure
    # ---------------------------------------------------
    output_filename = f"solve_times_{seed}.png"
    if solver_filter and solver_filter != "all":
        output_filename = f"solve_times_{seed}_{solver_filter}.png"

    output_path = os.path.join(directory, output_filename)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

    plt.show()

    # Print solver stats
    solver_stats = []
    for solver, df_list in solver_dfs.items():
        # Combine all CSV chunks for this solver
        combined_df = pd.concat(df_list, ignore_index=True)

        # "solved": rows where 'count_value' is not NaN
        solved_count = combined_df['count_value'].notna().sum()

        # "UNSATISFIABLE"
        unsat_count = combined_df[combined_df['satisfiability'] == 'UNSATISFIABLE'].shape[0]

        # "not solved": 'satisfiability' is empty/NaN AND 'count_value' is empty/NaN
        not_solved_count = combined_df[
            (combined_df['satisfiability'].isna() | (combined_df['satisfiability'] == '')) &
            (combined_df['count_value'].isna() | (combined_df['count_value'] == ''))
        ].shape[0]

        # "error == True"
        error_count = combined_df[combined_df['error'] == True].shape[0]

        # "timed_out == True"
        timed_out_count = combined_df[combined_df['timed_out'] == True].shape[0]

        solver_stats.append([
            solver,
            solved_count,
            unsat_count,
            not_solved_count,
            error_count,
            timed_out_count
        ])

    # Print results in a small table
    print("\nSolver Stats Summary:")
    print(f"{'Solver':<15} "
        f"{'Solved':<8} "
        f"{'UNSAT':<8} "
        f"{'Not-solved':<11} "
        f"{'Error':<8} "
        f"{'Timed Out':<10}")

    for (
        solver_name,
        solved_cnt,
        unsat_cnt,
        not_solved_cnt,
        error_cnt,
        timed_out_cnt
    ) in solver_stats:
        print(f"{solver_name:<15} "
            f"{solved_cnt:<8} "
            f"{unsat_cnt:<8} "
            f"{not_solved_cnt:<11} "
            f"{error_cnt:<8} "
            f"{timed_out_cnt:<10}")
        
def main():
    """
    Usage:
        python plot_times.py <seed> [solver_filter] [options]

    Examples:
        # Automatic scaling, all solvers:
        python plot_times.py s1779779729113289578

        # Automatic scaling, single solver:
        python plot_times.py s1779779729113289578 d4g

        # Set x-axis from 0..500, y-axis from 0..200
        python plot_times.py s1779779729113289578 d4g --xmin 0 --xmax 500 --ymin 0 --ymax 200

        # If single solver, y2-axis can be clamped via --y2min, --y2max as well
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", help="Seed string (e.g., s1779779729113289578)")
    parser.add_argument("solver_filter", nargs="?", default=None,
                        help="Specific solver (e.g., d4g, g4r) or 'all'. If omitted, defaults to all.")
    # Optional axis limits
    parser.add_argument("--xmin", type=float, default=None, help="Min X-axis (solve-time index) limit")
    parser.add_argument("--xmax", type=float, default=None, help="Max X-axis (solve-time index) limit")
    parser.add_argument("--ymin", type=float, default=None, help="Min Y-axis (solve-time) limit")
    parser.add_argument("--ymax", type=float, default=None, help="Max Y-axis (solve-time) limit")
    parser.add_argument("--y2min", type=float, default=None, help="Min Y2-axis (model-count) limit")
    parser.add_argument("--y2max", type=float, default=None, help="Max Y2-axis (model-count) limit")

    args = parser.parse_args()

    plot_solve_times(
        seed=args.seed,
        solver_filter=args.solver_filter,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        y2min=args.y2min,
        y2max=args.y2max
    )

if __name__ == "__main__":
    main()

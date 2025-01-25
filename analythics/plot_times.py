import sys
import os
import argparse
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_solve_times(seed, solver_filter=None,
                     xmin=None, xmax=None,
                     ymin=None, ymax=None,
                     y2min=None, y2max=None):
    directory = "../SharpVelvet/out/"
    files = [f for f in os.listdir(directory) if seed in f and f.endswith(".csv")]

    plt.rcParams.update({'font.size': 16})  # Global font size

    if not files:
        print(f"No files found for seed: {seed}")
        sys.exit(1)

    # Prepare a figure
    fig, ax1 = plt.subplots(figsize=(16, 8))  # Main axes for solve times

    solvers = {}
    solver_empty_counts = defaultdict(int)
    solver_errors = defaultdict(bool)
    solver_dfs = defaultdict(list)
    instance_counts = defaultdict(dict)

    # Process each CSV file
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        solver = file_name.split('_')[5].split('_fuzz-results')[0]

        if solver_filter and solver_filter != "all":
            if solver != solver_filter:
                continue

        df = pd.read_csv(file_path)
        df["instance_short"] = df["instance"].str.split("instances/").str[-1]

        if "error" in df.columns and df["error"].any():
            solver_errors[solver] = True

        if "count_value" in df.columns:
            df_empty_count = df["count_value"].isna().sum() + df["count_value"].eq("").sum()
            solver_empty_counts[solver] += df_empty_count

        df['primary_num'] = df['instance'].str.extract(r'_(\d{3})_')[0].astype(int)
        df['secondary_num'] = df['instance'].str.extract(r'_(\d{3}).cnf$')[0].astype(int)
        df['index'] = (df['primary_num'] + df['secondary_num'] * 10) / 10.1

        if "count_value" in df.columns:
            df['count_value'] = pd.to_numeric(df['count_value'], errors='coerce')

        solved = df[df["count_value"].notna()]
        unsolved = df[df["count_value"].isna()]

        solved_scatter = ax1.scatter(
            solved['index'],
            solved['solve_time'],
            s=6,
            marker='o',  # circle
            label=None
        )
        solvers[solver] = solved_scatter.get_facecolor()[0]

        ax1.scatter(
            unsolved['index'],
            unsolved['solve_time'],
            s=60,
            marker='x',  # circle with X
            color=solved_scatter.get_facecolor()[0],
            label=None
        )

        solver_dfs[solver].append(df)

        if (solver_filter is None) or (solver_filter == "all"):
            for idx, row in df.iterrows():
                inst = row["instance_short"]
                cval = row.get("count_value", None)
                if pd.notna(cval):
                    instance_counts[inst][solver] = str(int(float(cval)))

    if not solvers:
        print(f"No data found for solver filter '{solver_filter}'. Exiting.")
        return

    ax1.set_xlabel("Horn Clause Fraction", fontsize=16)
    ax1.set_ylabel("Solve Time (seconds)", fontsize=16)
    ax1.grid(True)

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

    if solver_filter and solver_filter != "all":
        combined_df = pd.concat(solver_dfs[solver_filter], ignore_index=True)
        combined_df = combined_df.dropna(subset=['count_value'])
        if not combined_df.empty:
            combined_df.sort_values(by='index', inplace=True)

            ax2 = ax1.twinx()
            ax2.set_ylabel("Model Count", fontsize=16)
            ax2.scatter(
                combined_df['index'],
                combined_df['count_value'],
                color='red',
                s=2,
                label="Model Count"
            )

            x_values = combined_df['index'].values
            y_values = combined_df['count_value'].values
            if len(x_values) > 3:
                coeffs = np.polyfit(x_values, y_values, deg=25)
                polynomial = np.poly1d(coeffs)

                x_smooth = np.linspace(x_values.min(), x_values.max(), 500)
                y_smooth = polynomial(x_smooth)

                ax2.plot(
                    x_smooth,
                    y_smooth,
                    color='red',
                    linewidth=1.5,
                    label="Model Count (Fitted)"
                )

            ax2.legend(loc="upper center", fontsize=16)

    if xmin is not None or xmax is not None:
        # Set x-axis limits
        ax1.set_xlim(left=xmin, right=xmax)

        # Filter data for the specified x-range
        filtered_solved = solved[(solved['index'] >= xmin) & (solved['index'] <= xmax)]
        filtered_unsolved = unsolved[(unsolved['index'] >= xmin) & (unsolved['index'] <= xmax)]
        
        # Automatically adjust y-axis limits based on the filtered data
        if ymin is None:
            ymin = min(filtered_solved['solve_time'].min(), filtered_unsolved['solve_time'].min())
        if ymax is None:
            ymax = max(filtered_solved['solve_time'].max(), filtered_unsolved['solve_time'].max())
        ax1.set_ylim(bottom=ymin, top=ymax)

        if solver_filter and solver_filter != "all" and not combined_df.empty:
            filtered_combined_df = combined_df[(combined_df['index'] >= xmin) & (combined_df['index'] <= xmax)]
            if y2min is None:
                y2min = filtered_combined_df['count_value'].min()
            if y2max is None:
                y2max = filtered_combined_df['count_value'].max()
            ax2.set_ylim(bottom=y2min, top=y2max)

    if ymin is not None or ymax is not None:
        ax1.set_ylim(bottom=ymin, top=ymax)

    if solver_filter and solver_filter != "all" and y2min is not None or y2max is not None:
        ax2.set_ylim(bottom=y2min, top=y2max)

    output_filename = f"solve_times_{seed}.png"
    if solver_filter and solver_filter != "all":
        output_filename = f"solve_times_{seed}_{solver_filter}.png"

    output_path = os.path.join(directory, output_filename)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", help="Seed string (e.g., s1779779729113289578)")
    parser.add_argument("solver_filter", nargs="?", default=None,
                        help="Specific solver (e.g., d4g, g4r) or 'all'. If omitted, defaults to all.")
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
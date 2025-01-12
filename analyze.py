import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def extract_last_three_digits(instance_path):
    match = re.search(r'_(\d{3})\.', instance_path)
    return int(match.group(1)) if match else None

def process_csv_files(input_dir, output_dir, seed=None):
    # Filter files by seed if provided
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv") and (seed in f if seed else True)]
    dataframes = {}
    mismatches = []

    for file in files:
        # Read and sort the CSV
        filepath = os.path.join(input_dir, file)
        df = pd.read_csv(filepath)
        df['last_three'] = df['instance'].apply(extract_last_three_digits)
        df = df.sort_values(by='last_three').drop(columns=['last_three'])
        output_path = os.path.join(output_dir, file)
        df.to_csv(output_path, index=False)
        print(f"Processed and saved sorted CSV: {output_path}")

        # Store for mismatch checks and plotting
        dataframes[file] = df

    # Check for mismatches
    base_file = list(dataframes.keys())[0]
    base_counts = dataframes[base_file][['instance', 'count_value']].set_index('instance')

    for file, df in dataframes.items():
        if file == base_file:
            continue
        merged = base_counts.join(df.set_index('instance')[['count_value']], how='inner', lsuffix='_base', rsuffix=f'_{file}')
        mismatch = merged[merged.iloc[:, 0] != merged.iloc[:, 1]]
        if not mismatch.empty:
            mismatches.append((file, mismatch))

    if mismatches:
        print("Mismatches detected:")
        for file, mismatch in mismatches:
            print(f"Mismatch in file {file}:")
            print(mismatch)
    else:
        print("All counts match.")

    return dataframes, mismatches

def plot_gpmc(dataframes):
    plt.figure(figsize=(12, 8))

    for file, df in dataframes.items():
        solver = re.search(r'_(\w+)_fuzz-results\.csv', file).group(1)
        indices = df['instance'].apply(lambda x: int(re.search(r'_(\d{3})\.', x).group(1)))

        if solver == 'gpmc':
            fig, ax1 = plt.subplots(figsize=(12, 8))

            ax1.set_xlabel("Instance Index")
            ax1.set_ylabel("Count Value", color='tab:blue')
            ax1.plot(indices, df['count_value'], label="gpmc (count)", marker='o', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax1.twinx()
            ax2.set_ylabel("Solving Time (seconds)", color='tab:orange')
            ax2.plot(indices, df['solve_time'], label="gpmc (solve time)", linestyle='--', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

            fig.tight_layout()
            plt.title("GPMC: Count Values vs Solving Times")
            plt.show()
            break

def plot_solving_times(dataframes):
    plt.figure(figsize=(12, 8))

    for file, df in dataframes.items():
        solver = re.search(r'_(\w+)_fuzz-results\.csv', file).group(1)
        indices = df['instance'].apply(lambda x: int(re.search(r'_(\d{3})\.', x).group(1)))

        plt.plot(indices, df['solve_time'], label=f"{solver} (solve time)", linestyle='--')

    plt.xlabel("Instance Index")
    plt.ylabel("Solving Time (seconds)")
    plt.title("Solver Solving Times")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def report_mismatches(mismatches):
    if not mismatches:
        print("All counts match.")
        plt.figure()
        plt.text(0.5, 0.5, "All counts match", fontsize=20, ha='center')
        plt.axis('off')
        plt.show()
    else:
        print("Mismatches detected:")
        for file, mismatch in mismatches:
            print(f"Mismatch in file {file}:")
            print(mismatch)

if __name__ == "__main__":
    input_directory = "SharpVelvet/out"
    output_directory = "SharpVelvet/out/sorted"
    seed = "s16320865262207757705"
    os.makedirs(output_directory, exist_ok=True)

    sorted_dataframes, mismatches = process_csv_files(input_directory, output_directory, seed=seed)
    plot_gpmc(sorted_dataframes)
    plot_solving_times(sorted_dataframes)
    report_mismatches(mismatches)

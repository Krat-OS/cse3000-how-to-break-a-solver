#!/bin/bash
#SBATCH --job-name="compile-generators"
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

# Fail fast on any error
set -e

# Dynamically determine the project directory based on the script's location
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Define absolute paths for project and configuration files
ENV_YAML="$PROJECT_DIR/env/global-env.yml"
GENERATE_INSTANCES_SCRIPT="$PROJECT_DIR/SharpVelvet/src/generate_instances.py"
RUN_FUZZER_SCRIPT="$PROJECT_DIR/SharpVelvet/src/run_fuzzer.py"
GENERATOR_CONFIG="$PROJECT_DIR/SharpVelvet/tool-config/generator_config_mc.json"
COUNTER_CONFIG="$PROJECT_DIR/SharpVelvet/tool-config/counter_config_mc.json"
INSTANCE_DIR="$PROJECT_DIR/SharpVelvet/out/instances"
CNF_DIR="$INSTANCE_DIR/cnf"
FEATURES_OUTPUT_DIR="$PROJECT_DIR/SharpVelvet/out/features_output"
SATZILLA_PATH="/home/vjurisic/revisiting_satzilla/SAT-features-competition2024/features"

# Conda environment name
CONDA_ENV_NAME="global-env"

# Activate Conda environment
eval "$(/home/vjurisic/miniconda3/bin/conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# Remove the existing instances directory
if [ -d "$INSTANCE_DIR" ]; then
    echo "Deleting existing instances directory: $INSTANCE_DIR"
    rm -rf "$INSTANCE_DIR"
else
    echo "No existing instances directory found."
fi

# Remove existing features output
if [ -d "$FEATURES_OUTPUT_DIR" ]; then
    echo "Deleting existing features output directory: $FEATURES_OUTPUT_DIR"
    rm -rf "$FEATURES_OUTPUT_DIR"
fi

mkdir -p "$FEATURES_OUTPUT_DIR"

# Ensure required Python packages are installed
REQUIRED_PACKAGES=("pandas")
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" &>/dev/null; then
        echo "Installing missing package: $package"
        pip install $package
    fi
done

# Generate instances without verifier
echo "Generating instances..."
python3 "$GENERATE_INSTANCES_SCRIPT" --generators "$GENERATOR_CONFIG"

# Run the solver (fuzzer) without cpog and verifier
echo "Running the solver..."
python3 "$RUN_FUZZER_SCRIPT" --counters "$COUNTER_CONFIG" --instances "$CNF_DIR"

# Compute features for generated instances
echo "Computing features for generated instances..."

# Remove existing CSV files in the features output directory
find "$FEATURES_OUTPUT_DIR" -type f -name "*.csv" -delete

# Process each instance
for CNF_FILE in "$CNF_DIR"/*.cnf; do
    if [ -f "$CNF_FILE" ]; then
        filename=$(basename "$CNF_FILE")
        generator_type="${filename%%_*}"
        OUTPUT_FILE="$FEATURES_OUTPUT_DIR/features_output_${generator_type}.csv"
        "$SATZILLA_PATH" -base "$CNF_FILE" "$OUTPUT_FILE" >/dev/null 2>&1
        echo "Computed features for $CNF_FILE"
    fi
done

# Post-process CSV files to remove duplicate rows
for CSV_FILE in "$FEATURES_OUTPUT_DIR"/*.csv; do
    if [ -f "$CSV_FILE" ]; then
        python3 - <<END
import pandas as pd
import os

csv_file_path = "$CSV_FILE"
if os.path.isfile(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path, header=None)
        first_row = df.iloc[0]
        df = df[df.ne(first_row).any(axis=1)]
        df.iloc[0] = first_row
        df.to_csv(csv_file_path, index=False, header=False)
    except Exception as e:
        print(f"Error processing {csv_file_path}: {e}")
END
        echo "Processed CSV file: $CSV_FILE"
    fi
done

# Deactivate Conda environment
conda deactivate
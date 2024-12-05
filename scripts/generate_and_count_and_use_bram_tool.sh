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
echo "PROJECT_DIR: $PROJECT_DIR"

# Define absolute paths for project and configuration files
ENV_YAML="$PROJECT_DIR/env/global-env.yml"
GENERATE_INSTANCES_SCRIPT="$PROJECT_DIR/SharpVelvet/src/generate_instances.py"
RUN_FUZZER_SCRIPT="$PROJECT_DIR/SharpVelvet/src/run_fuzzer.py"
GENERATOR_CONFIG="$PROJECT_DIR/SharpVelvet/tool-config/generator_config_mc.json"
COUNTER_CONFIG="$PROJECT_DIR/SharpVelvet/tool-config/counter_config_mc.json"
INSTANCE_DIR="$PROJECT_DIR/SharpVelvet/out/instances"
CNF_DIR="$INSTANCE_DIR/cnf"

# Paths for features extraction
FEATURES_OUTPUT_DIR="$PROJECT_DIR/SharpVelvet/out/features_output"
SATZILLA_PATH="$PROJECT_DIR/scripts/features"

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
        # Get the generator type from the filename
        filename=$(basename "$CNF_FILE")
        generator_type="${filename%%_*}"
        OUTPUT_FILE="$FEATURES_OUTPUT_DIR/features_output_${generator_type}.csv"

        # Run the SATzilla features extractor
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

# Ensure the CSV file exists before proceeding
csv_file_path = "$CSV_FILE"
if os.path.isfile(csv_file_path):
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path, header=None)

        # Keep the first row
        first_row = df.iloc[0]

        # Remove duplicate rows
        df = df[df.ne(first_row).any(axis=1)]

        # Reinsert the first row at the top
        df.iloc[0] = first_row

        # Save the updated CSV file
        df.to_csv(csv_file_path, index=False, header=False)
        print(f"Processed CSV file successfully: {csv_file_path}")
    except Exception as e:
        print(f"Error processing {csv_file_path}: {e}")
else:
    print(f"CSV file not found: {csv_file_path}")
END
        echo "Processed CSV file: $CSV_FILE"
    fi
done
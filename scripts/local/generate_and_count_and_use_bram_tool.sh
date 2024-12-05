#!/bin/bash

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Define absolute paths for project and configuration files
PROJECT_DIR="/home/parallels/RP/cse3000-how-to-break-a-solver"
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

# Conda environment name
CONDA_ENV_NAME="global-env"

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

# Check if the conda environment exists
if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "Conda environment '$CONDA_ENV_NAME' does not exist. Creating it..."
    conda env create -f "$ENV_YAML"
else
    echo "Conda environment '$CONDA_ENV_NAME' already exists."
fi

# Activate conda environment
conda activate "$CONDA_ENV_NAME"

# Ensure required Python packages are installed
REQUIRED_PACKAGES=("pandas")
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" &>/dev/null; then
        echo "Installing missing package: $package"
        pip install $package
    fi
done

# Generate instances without verifier
python3 "$GENERATE_INSTANCES_SCRIPT" --generators "$GENERATOR_CONFIG"

# Run the solver (fuzzer) without cpog and verifier
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
df = pd.read_csv("$CSV_FILE", header=None)
first_row = df.iloc[0]
df = df[df.ne(first_row).any(axis=1)]
df.iloc[0] = first_row
df.to_csv("$CSV_FILE", index=False, header=False)
END
        echo "Processed CSV file: $CSV_FILE"
    fi
done

# Deactivate conda environment
conda deactivate

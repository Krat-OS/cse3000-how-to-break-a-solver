#!/bin/bash

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Dynamically determine the project directory based on the script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")

ENV_YAML="$PROJECT_DIR/env/global-env.yml"
GENERATE_INSTANCES_SCRIPT="$PROJECT_DIR/SharpVelvet/src/generate_instances.py"
RUN_FUZZER_SCRIPT="$PROJECT_DIR/SharpVelvet/src/run_fuzzer.py"
GENERATOR_CONFIG="$PROJECT_DIR/SharpVelvet/tool-config/generator_config_mc.json"
COUNTER_CONFIG="$PROJECT_DIR/SharpVelvet/tool-config/counter_config_mc.json"
INSTANCE_DIR="$PROJECT_DIR/SharpVelvet/out/instances/cnf"

# Conda environment name
CONDA_ENV_NAME="global-env"

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
python3 "$RUN_FUZZER_SCRIPT" --counters "$COUNTER_CONFIG" --instances "$INSTANCE_DIR"

# Deactivate conda environment
conda deactivate

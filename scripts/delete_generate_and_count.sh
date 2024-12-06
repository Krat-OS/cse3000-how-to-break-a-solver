#!/bin/bash
#SBATCH --job-name="compile-generators"
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

module load miniconda3

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GENERATE_INSTANCES_SCRIPT="$PROJECT_DIR/SharpVelvet/src/generate_instances.py"
RUN_FUZZER_SCRIPT="$PROJECT_DIR/SharpVelvet/src/run_fuzzer.py"
GENERATOR_CONFIG="$PROJECT_DIR/SharpVelvet/tool-config/generator_config_mc.json"
COUNTER_CONFIG="$PROJECT_DIR/SharpVelvet/tool-config/counter_config_mc.json"
INSTANCE_DIR="$PROJECT_DIR/SharpVelvet/out/instances"

if [ -d "$INSTANCE_DIR" ]; then
    echo "Deleting existing instances directory: $INSTANCE_DIR"
    rm -rf "$INSTANCE_DIR"
else
    echo "No existing instances directory found."
fi

conda activate "global-env"
python3 "$GENERATE_INSTANCES_SCRIPT" --generators "$GENERATOR_CONFIG"
python3 "$RUN_FUZZER_SCRIPT" --counters "$COUNTER_CONFIG" --instances "$INSTANCE_DIR/cnf"
conda deactivate

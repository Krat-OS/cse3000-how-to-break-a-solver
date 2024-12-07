#!/bin/bash
#SBATCH --job-name="use-features"
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

# Delftblue
module load miniconda3

# Local
# eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)"

CNF_DIR="/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/out/instances/cnf"
FEATURES_OUTPUT_DIR="/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/out/features_output"
SATZILLA_PATH="/home/vjurisic/cse3000-how-to-break-a-solver/satzilla_feature_extractor/revisiting_satzilla_tool/SAT-features-competition2024/features"
FEATURES_CLI_SCRIPT="/home/$USER/cse3000-how-to-break-a-solver/satzilla_feature_extractor/compute_sat_feature_data_cli.py"

conda activate "global-env"
echo Done Conda Activation!

echo "Computing features for CNF instances..."
python3 $FEATURES_CLI_SCRIPT compute_features "$CNF_DIR" "$FEATURES_OUTPUT_DIR" "$SATZILLA_PATH"

echo "Processing CSV files..."
python3 $FEATURES_CLI_SCRIPT process_csv_files "$FEATURES_OUTPUT_DIR"

conda deactivate
echo "Done Conda Deactivation!"

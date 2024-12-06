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
# eval "$(/home/vjurisic/miniconda3/bin/conda shell.bash hook)"

CNF_DIR="/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/out/instances/cnf"
FEATURES_OUTPUT_DIR="/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/out/features_output"
SATZILLA_PATH="/home/$USER/cse3000-how-to-break-a-solver/revisiting_satzilla/SAT-features-competition2024/features"

# Remove existing features output directory if it exists
if [ -d "$FEATURES_OUTPUT_DIR" ]; then
    echo "Deleting existing features output directory: $FEATURES_OUTPUT_DIR"
    rm -rf "$FEATURES_OUTPUT_DIR"
else
    echo "No existing features output directory found."
fi

# Create a new features output directory
mkdir -p "$FEATURES_OUTPUT_DIR"

conda activate "global-env"
echo "Done Conda Activation!"

echo "Computing features for CNF instances..."

# Loop through all CNF files and compute features
for CNF_FILE in "$CNF_DIR"/*.cnf; do
    if [ -f "$CNF_FILE" ]; then
        filename=$(basename "$CNF_FILE")
        generator_type="${filename%%_*}"
        OUTPUT_FILE="$FEATURES_OUTPUT_DIR/features_output_${generator_type}.csv"
        "$SATZILLA_PATH" -base "$CNF_FILE" "$OUTPUT_FILE" >/dev/null 2>&1
        echo "Computed features for $CNF_FILE"
    fi
done

# Process CSV files to remove duplicates
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

conda deactivate
echo "Done Conda Deactivation!"

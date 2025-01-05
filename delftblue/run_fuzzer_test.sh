#!/bin/bash
#SBATCH --job-name="sharpvelvet-test-run"
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

# Load the Miniconda module
module load miniconda3
echo "Conda module loaded!"

# Activate the global-env environment
conda activate global-env
echo "Conda environment 'global-env' activated!"

# Run the Python script with the specified parameters
python /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/src/run_fuzzer.py \
  --counters /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/vuk/counter_config_gpmc.json \
  --instances /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/instances/1000_horn_instances/chevu

# Deactivate Conda environment
conda deactivate
echo "Conda environment 'global-env' deactivated!"


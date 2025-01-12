#!/bin/bash
#SBATCH --job-name="sharpvelvet-test-run-5solvers"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

# Load Miniconda module
module load miniconda3
echo "Conda module loaded!"

# Activate the Conda environment
conda activate global-env
echo "Conda environment 'global-env' activated!"

# Define the solver configuration files
configs=(
    "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/exp/d4g_counter_config_mc.json"
    "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/exp/d4r_counter_config_mc.json"
    "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/exp/ganak_counter_config_mc.json"
    "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/exp/gpmc_counter_config_mc.json"
)

# Run each solver configuration in parallel
for cfg in "${configs[@]}"; do
    python /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/src/run_fuzzer.py \
        --counters "$cfg" \
        --instances /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/instances/small-test \
        &
done

# Wait for all solver runs to complete
wait

# Deactivate the Conda environment
conda deactivate
echo "Conda environment 'global-env' deactivated!"

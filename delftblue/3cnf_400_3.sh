#!/bin/bash
#SBATCH --job-name="400-110-3cnf"
#SBATCH --time=12:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
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
    "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/exp/ganak_counter_config_mc.json"
    "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/exp/gpmc_counter_config_mc.json"
)

# Launch each solver configuration in parallel using srun
for i in "${!configs[@]}"; do
    srun --exclusive -n1 -c1 python /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/src/run_fuzzer.py \
        --counters "${configs[i]}" \
        --instances /home/vjurisic/cse3000-how-to-break-a-solver/SharpVelvet/instances/3cnf-400clause-110var-horn \
        &
done

# Wait for all solver runs to complete
wait

# Deactivate the Conda environment
conda deactivate
echo "Conda environment 'global-env' deactivated!"

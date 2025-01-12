#!/bin/bash
#SBATCH --job-name="sharpvelvet-test-run-5solvers"
#SBATCH --time=02:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

module load miniconda3
echo "Conda module loaded!"

conda activate global-env
echo "Conda environment 'global-env' activated!"

configs=(
    "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/exp/d4g_counter_config_mc.json"
    "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/exp/d4r_counter_config_mc.json"
    "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/exp/ganak_counter_config_mc.json"
    "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/exp/gpmc_counter_config_mc.json"
    "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/exp/sharpSAT_counter_config_mc.json"
)

for cfg in "${configs[@]}"; do
    srun --exclusive -n1 -c16 \
        python /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/src/run_fuzzer.py \
        --counters "$cfg" \
        --instances /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/instances/small-test \
    &
done

# Wait for all solver runs to finish
wait

conda deactivate
echo "Conda environment 'global-env' deactivated!"

#!/bin/bash
#SBATCH --job-name="compile-generators"
#SBATCH --time=00:04:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

# Delftblue
module load miniconda3

# Local
# eval "$(/home/vjurisic/miniconda3/bin/conda shell.bash hook)"

if [ -d "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/out/instances" ]; then
    echo "Deleting existing instances directory."
    rm -rf "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/out/instances"
fi

conda activate "global-env"
echo Done Conda Activation!
python3 "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/src/generate_instances.py" --generators "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/vuk/generator_config_mc.json"
python3 "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/src/run_fuzzer.py" --counters "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/vuk/counter_config_mc.json" --instances "/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/out/instances/cnf"
conda deactivate
echo Done Conda Deactivation!
#!/bin/bash
#SBATCH --job-name="sharpvelvet-run"
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

module load miniconda3
echo Done Conda loading!
conda activate sharpvelvet
echo Done Conda Activation!
python /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/src/run_fuzzer.py --counters /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/counter_config_mc.json --instances /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/out/instances/cnf
conda deactivate
echo Done Conda Deactivation!


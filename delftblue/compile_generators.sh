#!/bin/bash
#SBATCH --job-name="compile-generators"
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

module load 2024r1
module load gcc/13.2.0

cd /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/generators
g++ cnf-fuzz-biere.c -o biere-fuzz


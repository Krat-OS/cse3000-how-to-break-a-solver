#!/bin/bash
#SBATCH --job-name="sharpvelvet-compile"
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

module load 2024r1
module load gcc/13.2.0
module load miniconda3

mkdir -p /scratch/$USER/.conda
ln -s /scratch/$USER/.conda $HOME/.conda

cd /home/$USER/SharpVelvet-main/generators
g++ cnf-fuzz-biere.c -o biere-fuzz

conda env create -f /home/$USER/SharpVelvet-main/env/sharpvelvet.yml


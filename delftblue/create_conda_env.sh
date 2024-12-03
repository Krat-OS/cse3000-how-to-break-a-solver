#!/bin/bash
#SBATCH --job-name="create-conda-sharpvelvet"
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

module load miniconda3

mkdir -p /scratch/$USER/.conda
ln -s /scratch/$USER/.conda $HOME/.conda

conda env create -f /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet-main/env/sharpvelvet.yml


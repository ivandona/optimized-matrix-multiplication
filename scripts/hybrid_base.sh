#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:30:00
#PBS -q short_cpuQ
#PBS -o logs/outputs/hyb_o.txt
#PBS -e logs/outputs/hyb_e.txt

# Load the MPI module
module load mpich-3.2

mpirun.actual -n 4 ./optimized-matrix-multiplication/build/bin/algorithms/base/hybrid_base 1024 4 

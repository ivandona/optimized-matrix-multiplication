#!/bin/bash
#PBS -l select=1:ncpus=4:mem=1gb
#PBS -l walltime=00:30:00
#PBS -q short_cpuQ
#PBS -o logs/outputs/omp_base_o.txt
#PBS -e logs/outputs/omp_base_e.txt

./optimized-matrix-multiplication/build/bin/algorithms/base/omp_base 1024 4
#!/bin/bash

module load mpich-3.2

rm -f hybrid_base.sh.*

mpicc -g -Wall -fopenmp -o hybrid_base hybrid_base.c -std=c99
if [ $? -eq 0 ]; then
    qsub hybrid_base.sh
else
    echo "Compilation failed. Job submission aborted."
fi

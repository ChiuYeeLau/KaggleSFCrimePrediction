#!/bin/bash
#$ -N parallel
#$ -q free*,pub64
#$ -pe openmp 8-64
#$ -m beas

module load enthought_python
export OMP_NUM_THREADS=$CORES

python Ensemble2.py

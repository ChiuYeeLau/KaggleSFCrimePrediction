#!/bin/bash
#$ -N TEST
#$ -q free64
#$ -m beas

module load enthought_python
python Ensemble2.py

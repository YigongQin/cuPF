#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Longhorn v100 nodes
#
#   *** Single Serial Job in v100 Queue ***
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch sample.slurm" on a Longhorn login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#----------------------------------------------------
#SBATCH -J myjob           # Job name
#SBATCH -o conv_qoisasp3.o%j       # Name of stdout output file
#SBATCH -e myjob.e%j       # Name of stderr error file
#SBATCH -p v100-lm            # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 4               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 05:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=ygqin@utexas.edu
#SBATCH -A ASC21002       # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...
#conda activate mpi_py
module load python3
#export MY_SPECTRUM_OPTIONS="--gpu --aff on" 
module load launcher_gpu
module list
pwd
date

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=1aniso
#export LAUNCHER_JOB_FILE=video

#for G in 1 3 9 25 50
#for G in 1 50
#do
python3 ger.py 110  66 1
python3 ger.py 240 89 2
python3 ger.py 380 25 3
python3 ger.py 35  45 4

$LAUNCHER_DIR/paramrun
#done
#./phase_field Nucl_input line_AM/ 2
#$LAUNCHER_DIR/paramrun

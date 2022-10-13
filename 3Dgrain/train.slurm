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
#SBATCH -J 3Dgrain           # Job name
#SBATCH -o 3Dgrain.o%j       # Name of stdout output file
#SBATCH -e 3Dgrain.e%j       # Name of stderr error file
#SBATCH -p v100            # Queue (partition) name
#SBATCH -N 4               # Total # of nodes (must be 1 for serial)
#SBATCH -n 16               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 4:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=ygqin@utexas.edu
#SBATCH -A ASC21002       # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...
#export MY_SPECTRUM_OPTIONS="--gpu --aff on" 
module load launcher_gpu
module list
pwd
date

bash createcmd

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=cmdfile
$LAUNCHER_DIR/paramrun
#$LAUNCHER_DIR/paramrun

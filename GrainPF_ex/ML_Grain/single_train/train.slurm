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
#SBATCH -J final           # Job name
#SBATCH -o myjob.o%j       # Name of stdout output file
#SBATCH -e myjob.e%j       # Name of stderr error file
#SBATCH -p rtx            # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 4               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
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
export LAUNCHER_JOB_FILE=qoi_conv
#export LAUNCHER_JOB_FILE=video

#for G in 10 20 30 40 50 60 80 100 150 200 250 300 350 400 500
#do

#for G in 10 11 12 14 16 18 20 25 30 35 40 45 50 
for G in 60 80 100 120 140 

#170 200 
do

for R in 10 20 30 40 50 60 70 80 85 90 92 94 96 98 100
do
       #python3 _oldmulti_phase.py ${G} ${R}
       #python3 gen_input.py ${j}
       python3 ger.py ${G} ${R} 
       $LAUNCHER_DIR/paramrun
       #ibrun -n 1 ./phase_field Nucl_input line_AM/ ${j} 
done
done
#./phase_field Nucl_input line_AM/ 2
#$LAUNCHER_DIR/paramrun

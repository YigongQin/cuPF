# PF_cpp
Phase field simulation written in C++  
The codes are initially written in a class project EE382N. The codes are used for phase-field simulation for Aeolus project.  
Model comes from a nonlinear transformed Echebarria's model https://journals.aps.org/pre/abstract/10.1103/PhysRevE.70.061604  

Compiler: xl   
MPI: spectrum MPI (CUDA-aware)  

module load xl cuda spectrum_mpi  
export CUDA_PATH=$TACC_CUDA_DIR  
export MY_SPECTRUM_OPTIONS="--gpu --aff on" 

Compile: make  
run: line_model: ./phase_field INPUT_FILE MACRO_INPUTS  
     DNS: ibrun -n NUM_GPUS ./phase_field INPUT_FILE MACRO_INPUTS  
     
     
![Alt text](https://github.com/YigongQin/PF_cpp/edit/main/timing.png?raw=true  "Timing")

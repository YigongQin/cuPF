# PF_cpp
Phase field simulation written in C++  
The codes are initially written in a class project EE382N. The codes are used for phase-field simulation for Aeolus project.  
Model comes from a nonlinear transformed Echebarria's model https://journals.aps.org/pre/abstract/10.1103/PhysRevE.70.061604  

Compiler: xl   
MPI: spectrum MPI (CUDA-aware)  

module load xl cuda spectrum_mpi  
export CUDA_PATH=$TACC_CUDA_DIR  
export MY_SPECTRUM_OPTIONS="--gpu --aff on" 

Before running micro code, generate input files in MACRO_INPUTS_DIR:  
mkdir MACRO_INPUTS_DIR            (change the dir in the new_Ini_DNS.py file)  
python3 new_Ini_DNS.py MACRO_MAT_FILE

Compile: make  
line_model: ./phase_field INPUT_FILE MACRO_INPUTS_DIR  
DNS: ibrun -n NUM_GPUS ./phase_field INPUT_FILE MACRO_INPUTS_DIR 
     
     
<img width="556" alt="timing" src="https://user-images.githubusercontent.com/62076142/119079589-00022f00-b9be-11eb-837f-288778b5244c.png">
![scaling](https://user-images.githubusercontent.com/62076142/119079655-23c57500-b9be-11eb-844f-21b30837c56c.png)




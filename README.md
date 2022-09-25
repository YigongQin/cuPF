# PF_cpp
Phase field simulation written in C++ and CUDA 
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
     
MPI scaling
 
<img width="556" alt="timing" src="https://user-images.githubusercontent.com/62076142/119079589-00022f00-b9be-11eb-837f-288778b5244c.png">

<img width="556" alt="timing" src="https://user-images.githubusercontent.com/62076142/119079655-23c57500-b9be-11eb-844f-21b30837c56c.png">


Dentrite-scale simulation



https://user-images.githubusercontent.com/62076142/189384626-9093423b-6516-4eb5-9464-cf358a0a4ce4.mp4


Grain-scale simulation
![3Dview](https://user-images.githubusercontent.com/62076142/189384211-b82a2127-dd0f-4581-9a7e-67f8576419e9.png)

Please cite the paper https://www.sciencedirect.com/science/article/pii/S0927025622000660 if you are using the codes in this repository.



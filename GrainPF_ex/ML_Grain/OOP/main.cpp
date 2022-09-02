#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <map>
#include<functional>
#include<mpi.h>
#include <cmath>
#include <algorithm>
#include <iterator>
#include "params.h"
#include "PhaseField.h"
using namespace std;


int main(int argc, char** argv)
{
    // All the variables should be float (except for nx, ny, Mt, nts, ictype, which should be integer)

    // step 1 (input): read or calculate  parameters from "input"
    // and print out information: lxd, nx, ny, Mt
    // Create a text string, which is used to output the text file
    params_MPI pM;
//    MPI_Comm comm = MPI_COMM_WORLD;
//    MPI_Init(&argc, &argv);
//    MPI_Comm_rank(comm, &pM.rank);
//    MPI_Comm_size(comm, &pM.nproc);
    pM.rank=0;
    pM.nproc=1;
     
    //if rank == 0: print('GPUs on this node', cuda.gpus)
    //printf('device id',gpu_name,'host id',rank );

    pM.nprocx = (int) ceil(sqrt(pM.nproc));
    pM.nprocy = (int) ceil(pM.nproc/pM.nprocx);
    pM.px = pM.rank%pM.nprocx;           //# x id of current processor   [0:nprocx]
    pM.py = pM.rank/pM.nprocx; //# y id of current processor  [0:nprocy]
    printf("px, %d, py, %d, for rank %d \n",pM.px, pM.py, pM.rank);

    if (pM.rank ==0){ printf("total/x/y processors %d, %d, %d\n", pM.nproc, pM.nprocx, pM.nprocy);}



    char* fileName=argv[1]; 
    string mac_folder = argv[2];


    PhaseField* pf_solver; // initialize the pointer to the class
    pf_solver = new PhaseField();
    pf_solver->mac.folder = mac_folder;
    pf_solver->parseInputParams(mac, fileName);
    pf_solver->q->num_case = 1;  //set parameters of realizations
    pf_solver->q->valid_run = 1; 
    pf_solver->cpuSetup(pM);
    pf_solver->params.seed_val = atoi(argv[3]);
    


 //   float* frac_ini = (float*) malloc(params.num_theta* sizeof(float));
 //   float sum_frac = 0.0;
 //   int* grain_grid = (int*) malloc(params.num_theta* sizeof(int)); 
  //  std::default_random_engine generator;
 //   std::normal_distribution<float> distribution(grain_size,0.35*grain_size);

    // start the region of gathering lots of runs
    for (int run=0;run<pf_solver->q->num_case;run++){
   
    printf("case %d\n",run);
   // int loc_seed = 20*((int)G0) + (int) (10000*Rmax);
   // loc_seed = loc_seed*num_case + run;
   // srand(loc_seed);
   // generator.seed( loc_seed );
   // int* aseq=(int*) malloc(params.num_theta* sizeof(int));
   // initialize the angles for every PF, while keep the liquid 0 
   // for (int i=0; i<params.num_theta; i++){
    //   aseq[i] = i+1;} //rand()%NUM_PF +1;

    pf_solver->initField(mac);
    pf_solver->cudaSetup(pM);
    pf_solver->evolve(mac);

  //  if (run>=num_case-valid_run){
   //     int loca_case = run-(num_case-valid_run);
    //       memcpy(alpha_asse+loca_case*full_length, pf_solver->alpha_i_full,sizeof(int)*full_length);

   // }   

    }

    pf_solver->output(pM);

    //MPI_Finalize();

    return 0;
}

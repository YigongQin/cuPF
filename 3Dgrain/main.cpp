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
#include "APTPhaseField.h"
using namespace std;


int main(int argc, char** argv)
{
    params_MPI pM;
//    MPI_Comm comm = MPI_COMM_WORLD;
//    MPI_Init(&argc, &argv);
//    MPI_Comm_rank(comm, &pM.rank);
//    MPI_Comm_size(comm, &pM.nproc);
    pM.rank=0;
    pM.nproc=1;

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
    pf_solver->parseInputParams(fileName);
    pf_solver->q->num_case = 1;  //set parameters of realizations
    pf_solver->q->valid_run = 1; 
    pf_solver->cpuSetup(pM);
    pf_solver->params.seed_val = atoi(argv[3]);

    // start the region of gathering lots of runs
    for (int run=0;run<pf_solver->q->num_case;run++){
   
    printf("case %d\n",run);
    pf_solver->initField();
    pf_solver->cudaSetup(pM);
    pf_solver->evolve();

    }

    pf_solver->output(pM);

    //MPI_Finalize();

    return 0;
}

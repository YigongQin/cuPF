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
#include <assert.h>
#include "params.h"
#include "PhaseField.h"
#include "APTPhaseField.h"
#include "QOI.h"
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


    int opt;
    string inputfile = argv[1]; 
    string mac_folder = "forcing/case";
    string out_folder = "/scratch1/07428/ygqin/graph/"; 
    bool APTon = true;
    bool checkCorrectness = false;
    int seed_val;
    static struct option long_options[] = {
        {"help",     0, 0,  '?'},
        {"check",    0, 0,  'c'},
        {"bench",    1, 0,  'b'},
        {"macfile",  1, 0,  'f'},
        {"APTon",    1, 0,  'a'},
        {"seed",     1, 0,  's'},
        {"output",   1, 0,  'o'},
        {0 ,0, 0, 0}
    };

    
    while ((opt = getopt_long(argc, argv, "b:f:o:a:s:c?", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'c':
            checkCorrectness = true;
            break;
        case 'f':
            mac_folder = optarg;
            break;
        case 'a':
            APTon = false;
            break;
        case 's':
            seed_val = atoi(optarg);
            break;
        case 'o':
            out_folder = out_folder + optarg;
            break;
        }
    }

    // run the python initialization for temperature and orientation field
    string cmd = "python3 " + inputfile + " " + to_string(seed_val);
    int result = system(cmd.c_str()); 
    assert(result == 0);

    PhaseField* pf_solver;
    if (APTon){
        printf("use APT algorithm\n");
        pf_solver = new APTPhaseField();
    }else{
    pf_solver = new PhaseField();
    }
    pf_solver->mac.folder = mac_folder + to_string(seed_val);
    pf_solver->out_folder = out_folder;
    pf_solver->params.seed_val = seed_val;
    pf_solver->parseInputParams(inputfile);
    pf_solver->q = new QOI();
    pf_solver->q->num_case = 1;  //set parameters of realizations
    pf_solver->q->valid_run = 1; 
    pf_solver->cpuSetup(pM);
   // pf_solver->params.seed_val = seed_val;

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

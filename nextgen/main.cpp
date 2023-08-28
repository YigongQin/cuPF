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
#include <iostream>
#include "params.h"
#include "PhaseField.h"
#include "APTPhaseField.h"
#include "QOI.h"
#include "thermalInputData.h"
using namespace std;


int main(int argc, char** argv)
{
    MPIsetting mpiManager;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &mpiManager.rank);
    MPI_Comm_size(comm, &mpiManager.numProcessor);
    mpiManager.twoDimensionPartition();
    mpiManager.printMPIsetting();
    // mpiManager.rank=0;
    // mpiManager.nproc=1;

    string inputFile = argv[1]; 
    string thermalInputFolder = "forcing/case";
    // this output folder should be specified differently for each system
    string outputFolder = "/scratch/07428/ygqin/graph/"; 

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

    int opt;    
    bool useAPT = true;
    bool checkCorrectness = false;
    bool save3DField = false;
    int seedValue;

    while ((opt = getopt_long(argc, argv, "b:f:o:a:s:c?", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'c':
            checkCorrectness = true;
            break;
        case 'f':
            thermalInputFolder = optarg;
            break;
        case 'a':
            useAPT = false;
            break;
        case 's':
            seedValue = atoi(optarg);
            break;
        case 'o':
            outputFolder = outputFolder + optarg;
            break;
        case 'b':
            save3DField = true;    
            cout << "save entire 3D data" << endl;
        }
    }

    // run the python initialization for temperature and orientation field
    string cmd = "python3 " + inputFile + " " + to_string(seedValue);
    int result = system(cmd.c_str()); 
    assert(result == 0);

    PhaseField* PFSolver;
    if (useAPT == true)
    {
        PFSolver = new APTPhaseField();
        cout << "use active parameter tracking algorithm" << endl;
    }
    else
    {
        PFSolver = new PhaseField();
        cout << "use full PF" << endl;
    }

    PFSolver->mac.folder = thermalInputFolder + to_string(seedValue);

    PFSolver->params.seed_val = seedValue;
    PFSolver->parseInputParams(inputFile);
    
    PFSolver->cpuSetup(mpiManager);
    PFSolver->qois = new QOI(PFSolver->params);

    PFSolver->initField();
    PFSolver->cudaSetup(mpiManager);
    PFSolver->evolve();
    PFSolver->output(mpiManager, outputFolder, save3DField);

    MPI_Finalize();

    return 0;
}

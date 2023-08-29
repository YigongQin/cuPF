#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include<functional>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <assert.h>
#include <iostream>
#include "params.h"
#include "PhaseField.h"
#include "APTPhaseField.h"
#include "QOI.h"
#include "ThermalInputData.h"
#include "MPIsetting.h"

using namespace std;


int main(int argc, char** argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);

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
        {"mpiDim",   1, 0,  'm'},
        {0 ,0, 0, 0}
    };

    int opt;    
    bool useAPT = true;
    bool checkCorrectness = false;
    bool save3DField = false;
    int seedValue;
    int mpiDim = 1;

    while ((opt = getopt_long(argc, argv, "b:f:o:a:s:c:m?", long_options, NULL)) != EOF) {

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

    MPIsetting* mpiManager;
    if (mpiDim == 1)
    {
         mpiManager = new MPIsetting1D(comm);
    }
    else if (mpiDim == 2)
    {
         mpiManager = new MPIsetting2D(comm);
    }
    
    MPI_Comm_rank(comm, &mpiManager->rank);
    MPI_Comm_size(comm, &mpiManager->numProcessor);
    mpiManager->domainPartition();
    mpiManager->printMPIsetting();

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

    PFSolver->SetMPIManager(mpiManager);

    PFSolver->initField();
    PFSolver->cudaSetup();
    PFSolver->evolve();
    PFSolver->output(outputFolder, save3DField);

    MPI_Finalize();

    return 0;
}

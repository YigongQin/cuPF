#include <string>
#include <assert.h>
#include <iostream>
#include "DesignSettingData.h"
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
    
    DesignSettingData* designSetting = new DesignSettingData();
    designSetting->getOptions(argc, argv);

    MPIsetting* mpiManager;
    if (designSetting->mpiDim == 1)
    {
         mpiManager = new MPIsetting1D(comm);
    }
    else if (designSetting->mpiDim == 2)
    {
         mpiManager = new MPIsetting2D(comm);
    }
    
    MPI_Comm_rank(comm, &mpiManager->rank);
    MPI_Comm_size(comm, &mpiManager->numProcessor);
    mpiManager->domainPartition();
    mpiManager->printMPIsetting();

    PhaseField* PFSolver;
    if (designSetting->useAPT == true)
    {
        PFSolver = new APTPhaseField();
        cout << "use active parameter tracking algorithm" << endl;
    }
    else
    {
        PFSolver = new PhaseField();
        cout << "use full PF" << endl;
    }

    PFSolver->SetDesignSetting(designSetting);
    PFSolver->SetMPIManager(mpiManager);
    PFSolver->mac.folder = designSetting->thermalInputFolder + to_string(0);

    PFSolver->params.seed_val = designSetting->seedValue;
    PFSolver->parseInputParams(designSetting->inputFile);
    PFSolver->cpuSetup(mpiManager);

    if (designSetting->useLineConfig == true)
    {
        PFSolver->qois = new QOILine(PFSolver->params);
    }
    else
    {
        PFSolver->qois = new QOI3D(PFSolver->params);
    }
    
    cout << "field initialization on cpu" <<endl;
    PFSolver->initField();
    cout << "cuda setup" <<endl;
    PFSolver->cudaSetup();
    PFSolver->evolve();
   // PFSolver->OutputQoIs();

    MPI_Finalize();

    return 0;
}

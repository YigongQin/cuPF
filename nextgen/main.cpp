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

    // run the python initialization for temperature and orientation field
    if (mpiManager->rank == 0)
    {
        string cmd = "python3 " + designSetting->inputFile + " " + to_string(designSetting->seedValue);
        int result = system(cmd.c_str()); 
        assert(result == 0);
    }

    PFSolver->SetDesignSetting(designSetting);
    PFSolver->mac.folder = designSetting->thermalInputFolder + to_string(designSetting->seedValue);

    PFSolver->params.seed_val = designSetting->seedValue;
    PFSolver->parseInputParams(designSetting->inputFile);
    PFSolver->cpuSetup(mpiManager);
    PFSolver->qois = new QOI(PFSolver->params);

    PFSolver->SetMPIManager(mpiManager);

    cout << "field initialization on cpu" <<endl;
    PFSolver->initField();
    cout << "cuda setup" <<endl;
    PFSolver->cudaSetup();
    PFSolver->evolve();
    PFSolver->output(designSetting->outputFolder, designSetting->save3DField);

    MPI_Finalize();

    return 0;
}

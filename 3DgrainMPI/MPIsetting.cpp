//////////////////////
// MPIsetting.cpp - MPI setting class implementation
// COPYRIGHT Yigong Qin, ALL RIGHTS RESERVED
/////////////////////

#include "MPIsetting.h"
#include <math.h>
#include <iostream>

using namespace std;

MPIsetting::MPIsetting(MPI_Comm commWorld) : comm(commWorld)
{
}

void MPIsetting::calculateTransferDataSize(int numFields)
{
    mNumFields = numFields;
    dataSizeX = numFields*haloWidth*nyLocal*nzLocal;
    dataSizeY = numFields*haloWidth*nxLocal*nzLocal;
    dataSizeXY = numFields*haloWidth*haloWidth*nzLocal;
}

void MPIsetting::printMPIsetting() const
{
    if (rank == 0)
    {
        cout<< "total/x/y/z number of processors: " << numProcessor << "/" << numProcessorX << "/" << numProcessorY << "/" << numProcessorZ << endl;
    }
    cout<< "rank/x/y/z processor IDs: " << rank << "/" << processorIDX << "/" << processorIDY << "/" << processorIDZ << endl;
}

void MPIsetting1D::domainPartition()
{
	numProcessorX = numProcessor;
    numProcessorY = 1;
    numProcessorZ = 1;
    processorIDX = rank;  
    processorIDY = 0; 
    processorIDZ = 0;
}

void MPIsetting2D::domainPartition()
{
	numProcessorX = (int) ceil(sqrt(numProcessor));
    numProcessorY = (int) ceil(numProcessor/numProcessorX);
    numProcessorZ = 1;
    processorIDX = rank%numProcessorX;  
    processorIDY = rank/numProcessorX; 
    processorIDZ = 0;
}


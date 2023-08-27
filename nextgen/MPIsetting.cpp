#include "MPIsetting.h"
#include <math.h>
#include <iostream>
using namespace std;

void MPIsetting::twoDimensionPartition()
{
	numProcessorX = (int) ceil(sqrt(numProcessor));
    numProcessorY = (int) ceil(numProcessor/numProcessorX);
    numProcessorZ = 1;
    processorIDX = rank%numProcessorX;  
    processorIDY = rank/numProcessorX; 
    processorIDZ = 0;
}

void MPIsetting::printMPIsetting() const
{
    if (rank == 0)
    {
        cout<< "total/x/y/z number of processors: " << numProcessor << "/" << numProcessorX << "/" << numProcessorY << "/" << numProcessorZ << endl;
    }
    cout<< "rank/x/y/z processor IDs: " << rank << "/" << processorIDX << "/" << processorIDY << "/" << processorIDZ << endl;
}

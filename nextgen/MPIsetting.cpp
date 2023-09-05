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
    int dataSizeX = numFields*haloWidth*nyLocal*nzLocal;
    int dataSizeY = numFields*haloWidth*nxLocal*nzLocal;
    int dataSizeXY = numFields*haloWidth*haloWidth*nzLocal;
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

template <typename T> std::map<std::string, std::pair<T*, int> > 
MPIsetting1D::createBoundaryBuffer(int numFields)
{
    std::map<std::string, std::pair<T*, int> > mMPIBuffer;
    mGeometrySize.push_back(haloWidth*nyLocal*nzLocal);
    calculateTransferDataSize(numFields);
    mMPIBuffer.emplace("sendL", std::make_pair(new T, dataSizeX));
    mMPIBuffer.emplace("sendR", std::make_pair(new T, dataSizeX));
    mMPIBuffer.emplace("recvL", std::make_pair(new T, dataSizeX));
    mMPIBuffer.emplace("recvR", std::make_pair(new T, dataSizeX));
    return mMPIBuffer;
}

template <typename T>
void MPIsetting1D::exchangeBoundaryData(int nTimeStep, std::map<std::string, std::pair<T*, int> > mMPIBuffer)
{
    int ntag = 8*nTimeStep;
    if ( processorIDX < numProcessorX-1 ) 
    {
        MPI_Send(mMPIBuffer["sendR"].first, mMPIBuffer["sendR"].second, T, rank+1, ntag+1, comm);
        MPI_Recv(mMPIBuffer["recvR"].first, mMPIBuffer["recvR"].second, T, rank+1, ntag+2, comm, MPI_STATUS_IGNORE);
    }
    if ( processorIDX > 0 )
    {
        MPI_Recv(mMPIBuffer["recvL"].first, mMPIBuffer["recvL"].second, T, rank-1, ntag+1, comm, MPI_STATUS_IGNORE);
        MPI_Send(mMPIBuffer["sendL"].first, mMPIBuffer["sendL"].second, T, rank-1, ntag+2, comm);
    }    
}

template <typename T> std::map<std::string, std::pair<T*, int> >  
MPIsetting2D::createBoundaryBuffer(int numFields)
{
    std::map<std::string, std::pair<T*, int> > mMPIBuffer;
    mGeometrySize.push_back(haloWidth*nyLocal*nzLocal);
    mGeometrySize.push_back(haloWidth*nxLocal*nzLocal);
    mGeometrySize.push_back(haloWidth*haloWidth*nzLocal);

    calculateTransferDataSize(numFields);
    mMPIBuffer.emplace("sendL", std::make_pair(new T, dataSizeX));
    mMPIBuffer.emplace("sendR", std::make_pair(new T, dataSizeX));
    mMPIBuffer.emplace("recvL", std::make_pair(new T, dataSizeX));
    mMPIBuffer.emplace("recvR", std::make_pair(new T, dataSizeX));

    mMPIBuffer.emplace("sendT", std::make_pair(new T, dataSizeY));
    mMPIBuffer.emplace("sendB", std::make_pair(new T, dataSizeY));
    mMPIBuffer.emplace("recvT", std::make_pair(new T, dataSizeY));
    mMPIBuffer.emplace("recvB", std::make_pair(new T, dataSizeY));  

    mMPIBuffer.emplace("sendLT", std::make_pair(new T, dataSizeXY));
    mMPIBuffer.emplace("sendRT", std::make_pair(new T, dataSizeXY));
    mMPIBuffer.emplace("recvLT", std::make_pair(new T, dataSizeXY));
    mMPIBuffer.emplace("recvRT", std::make_pair(new T, dataSizeXY));
    mMPIBuffer.emplace("sendLB", std::make_pair(new T, dataSizeXY));
    mMPIBuffer.emplace("sendRB", std::make_pair(new T, dataSizeXY));
    mMPIBuffer.emplace("recvLB", std::make_pair(new T, dataSizeXY));
    mMPIBuffer.emplace("recvRB", std::make_pair(new T, dataSizeXY));     
    
    return mMPIBuffer;
}

template <typename T>
void MPIsetting2D::exchangeBoundaryData(int nTimeStep, std::map<std::string, std::pair<T*, int> > mMPIBuffer)
{
    int ntag = 8*nTimeStep;
    if ( processorIDX < numProcessorX-1 ) 
    {
        MPI_Send(mMPIBuffer["sendR"].first, dataSizeX, T, rank+1, ntag+1, comm);
        MPI_Recv(mMPIBuffer["recvR"].first, dataSizeX, T, rank+1, ntag+2, comm, MPI_STATUS_IGNORE);
    }
    if ( processorIDX > 0 )
    {
        MPI_Recv(mMPIBuffer["recvL"].first, dataSizeX, T, rank-1, ntag+1, comm, MPI_STATUS_IGNORE);
        MPI_Send(mMPIBuffer["sendL"].first, dataSizeX, T, rank-1, ntag+2, comm);
    }

    if ( processorIDY < numProcessorY-1 ) 
    {
        MPI_Send(mMPIBuffer["sendT"].first, dataSizeY, T, rank+numProcessorX, ntag+3, comm);
        MPI_Recv(mMPIBuffer["recvT"].first, dataSizeY, T, rank+numProcessorX, ntag+4, comm, MPI_STATUS_IGNORE);
    }
    if ( processorIDY > 0 )
    {
        MPI_Recv(mMPIBuffer["recvB"].first, dataSizeY, T, rank-numProcessorX, ntag+3, comm, MPI_STATUS_IGNORE);
        MPI_Send(mMPIBuffer["sendB"].first, dataSizeY, T, rank-numProcessorX, ntag+4, comm);
    }

    if ( processorIDX < numProcessorX-1 and processorIDY < numProcessorY-1)
    {
        MPI_Send(mMPIBuffer["sendRT"].first, dataSizeXY, T, rank+1+numProcessorX, ntag+5, comm);
        MPI_Recv(mMPIBuffer["recvRT"].first, dataSizeXY, T, rank+1+numProcessorX, ntag+6, comm, MPI_STATUS_IGNORE);
    }
    if ( processorIDX > 0 and processorIDY > 0 )
    {
        MPI_Recv(mMPIBuffer["recvLB"].first, dataSizeXY, T, rank-1-numProcessorX, ntag+5, comm, MPI_STATUS_IGNORE);
        MPI_Send(mMPIBuffer["sendLB"].first, dataSizeXY, T, rank-1-numProcessorX, ntag+6, comm);
    }

    if ( processorIDY < numProcessorY-1 and processorIDX > 0 )
    {
        MPI_Send(mMPIBuffer["sendLT"].first, dataSizeXY, T, rank+numProcessorX-1, ntag+7, comm);
        MPI_Recv(mMPIBuffer["recvLT"].first, dataSizeXY, T, rank+numProcessorX-1, ntag+8, comm, MPI_STATUS_IGNORE);
    }
    if ( processorIDY>0 and processorIDX < numProcessorX-1 )
    {
        MPI_Recv(mMPIBuffer["recvRB"].first, dataSizeXY, T, rank-numProcessorX+1, ntag+7, comm, MPI_STATUS_IGNORE);
        MPI_Send(mMPIBuffer["sendRB"].first, dataSizeXY, T, rank-numProcessorX+1, ntag+8, comm);
    }
}
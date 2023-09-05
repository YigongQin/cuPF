//////////////////////
// MPIsetting.h
// COPYRIGHT Yigong Qin, ALL RIGHTS RESERVED
/////////////////////

#pragma once

#include <mpi.h>
#include <string>
#include <map>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

class MPIsetting
{
public:
    MPIsetting(MPI_Comm commWorld);
	void printMPIsetting() const;
    void calculateTransferDataSize(int numFields);
	virtual void domainPartition() = 0;

    MPI_Comm comm;
    int rank, numProcessor;
    int processorIDX, processorIDY, processorIDZ;
    int numProcessorX, numProcessorY, numProcessorZ;
    int nxLocal, nyLocal, nzLocal, nzLocalAll;
    int haloWidth = 1;
    int mNumFields;
    std::vector<int> mGeometrySize;

    std::map<std::string, std::pair<float*, int> > PFBuffer; 
    std::map<std::string, std::pair<int*, int> > ArgBuffer;
    std::vector<std::pair<float*, int> > data_old_float;
    std::vector<std::pair<float*, int> > data_new_float;
    std::vector<std::pair<int*, int> > data_old_int;
    std::vector<std::pair<int*, int> > data_new_int;

protected:
    int ntag, dataSizeX, dataSizeY, dataSizeXY; 

};

class MPIsetting1D : public MPIsetting
{
public:
    MPIsetting1D(MPI_Comm commWorld) : MPIsetting(commWorld) {};
    void domainPartition() override;
    template <typename T> std::map<std::string, std::pair<T*, int> > createBoundaryBuffer(int numFields);
    template <typename T, typename U> void exchangeBoundaryData(int nTimeStep, std::map<std::string, std::pair<U*, int> > mMPIBuffer);
    void MPItransferData(int nTimeStep, std::vector<std::pair<float*, int> > fields, std::map<std::string, std::pair<float*, int> > mMPIBuffer);
    void MPItransferData(int nTimeStep, std::vector<std::pair<int*, int> > fields, std::map<std::string, std::pair<int*, int> > mMPIBuffer);
};

class MPIsetting2D : public MPIsetting
{
public:
    MPIsetting2D(MPI_Comm commWorld) : MPIsetting(commWorld) {};
    void domainPartition() override;
    template <typename T> std::map<std::string, std::pair<T*, int> > createBoundaryBuffer(int numFields);
    template <typename T, typename U> void exchangeBoundaryData(int nTimeStep, std::map<std::string, std::pair<U*, int> > mMPIBuffer);
};


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

template <typename T, typename U>
void MPIsetting1D::exchangeBoundaryData(int nTimeStep, std::map<std::string, std::pair<U*, int> > mMPIBuffer)
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

template <typename T, typename U>
void MPIsetting2D::exchangeBoundaryData(int nTimeStep, std::map<std::string, std::pair<U*, int> > mMPIBuffer)
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



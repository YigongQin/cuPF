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
    virtual void createBoundaryBuffer(int numFields) = 0;
    virtual void exchangeBoundaryData(int nTimeStep) = 0;
    void MPItransferData(int nTimeStep, std::vector<std::pair<float*, int>> fields);

    MPI_Comm comm;
    int rank, numProcessor;
    int processorIDX, processorIDY, processorIDZ;
    int numProcessorX, numProcessorY, numProcessorZ;
    int nxLocal, nyLocal, nzLocal, nzLocalAll;
    int haloWidth = 1;
    int mNumFields;
    std::vector<int> mGeometrySize;
    std::map<std::string, std::pair<float*, int> > mMPIBuffer;

protected:
    int ntag, dataSizeX, dataSizeY, dataSizeXY; 
};

class MPIsetting1D : public MPIsetting
{
public:
    MPIsetting1D(MPI_Comm commWorld) : MPIsetting(commWorld) {};
    void domainPartition() override;
    void createBoundaryBuffer(int numFields) override;
    void exchangeBoundaryData(int nTimeStep) override;
};

class MPIsetting2D : public MPIsetting
{
public:
    MPIsetting2D(MPI_Comm commWorld) : MPIsetting(commWorld) {};
    void domainPartition() override;
    void createBoundaryBuffer(int numFields) override;
    void exchangeBoundaryData(int nTimeStep) override;
};


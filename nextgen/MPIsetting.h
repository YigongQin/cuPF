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
    template <typename T> void exchangeBoundaryData(int nTimeStep, std::map<std::string, std::pair<T*, int> > mMPIBuffer);
    template <typename T> void MPItransferData(int nTimeStep, std::vector<std::pair<T*, int> > fields, 
                                               std::map<std::string, std::pair<T*, int> > mMPIBuffer);
};

class MPIsetting2D : public MPIsetting
{
public:
    MPIsetting2D(MPI_Comm commWorld) : MPIsetting(commWorld) {};
    void domainPartition() override;
    template <typename T> std::map<std::string, std::pair<T*, int> > createBoundaryBuffer(int numFields);
    template <typename T> void exchangeBoundaryData(int nTimeStep, std::map<std::string, std::pair<T*, int> > mMPIBuffer);
};


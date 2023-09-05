#include "MPIsetting.h"
#include "devicefunc.cu_inl"
#include <algorithm>


template <typename T> __global__ void
collectData1D(float* field, int numFields, int offset, T* sendBufferL, T* sendBufferR, int nxLocal, int nyLocal, int nzLocal, int haloWidth)
{
    int C = blockIdx.x * blockDim.x + threadIdx.x;
    int i, j, k, PF_id, fnx, fny, fnz;
    G2L_3D(C, i, j, k, PF_id, nxLocal, nyLocal, nzLocal);
    if ( (i<haloWidth) && (j<nyLocal) && (k<nzLocal) && (PF_id<numFields))
    {
        fnx = nxLocal + haloWidth*2;
        fny = nyLocal + haloWidth*2;
        fnz = nzLocal + haloWidth*2;

        int field_indexL = L2G_4D(i + haloWidth, j + haloWidth, k + haloWidth, PF_id, fnx, fny, fnz);
        int field_indexR = L2G_4D(i + nxLocal, j + haloWidth, k + haloWidth, PF_id, fnx, fny, fnz);

        sendBufferL[C + offset] = field[field_indexL];
        sendBufferR[C + offset] = field[field_indexR];
    }
}

template <typename T> __global__ void
distributeData1D(float* field, int numFields, int offset, T* recvBufferL, T* recvBufferR, int nxLocal, int nyLocal, int nzLocal, int haloWidth)
{
  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j, k, PF_id, fnx, fny, fnz;
  G2L_3D(C, i, j, k, PF_id, nxLocal, nyLocal, nzLocal);

  if ( (i<haloWidth) && (j<nyLocal) && (k<nzLocal) && (PF_id<numFields))
  {
      fnx = nxLocal + haloWidth*2;
      fny = nyLocal + haloWidth*2;
      fnz = nzLocal + haloWidth*2;

      int field_indexL = L2G_4D(i, j + haloWidth, k + haloWidth, PF_id, fnx, fny, fnz);
      int field_indexR = L2G_4D(i + nxLocal + haloWidth, j + haloWidth, k + haloWidth, PF_id, fnx, fny, fnz);

      field[field_indexL] = recvBufferL[C + offset];
      field[field_indexR] = recvBufferR[C + offset];
  }
}



void MPIsetting1D::MPItransferData(int nTimeStep, std::vector<std::pair<int*, int>> fieldChunks, std::map<std::string, std::pair<int*, int> > mMPIBuffer)
{
    int blocksize_2d = 128;  // seems reduce the block size makes it a little faster, but around 128 is okay.
    for (auto & field : fieldChunks)
    {
        int threadsRequired = *std::max_element(mGeometrySize.begin(), mGeometrySize.end())*field.second;
        int dataAcquired = 0;
        int num_block_2d = (threadsRequired + blocksize_2d -1)/blocksize_2d;
        collectData1D<T><<< num_block_2d, blocksize_2d >>>(field.first, field.second, dataAcquired, mMPIBuffer["sendL"].first, mMPIBuffer["sendR"].first,
                                                        nxLocal, nyLocal, nzLocal, haloWidth);
        dataAcquired += field.second*mGeometrySize[0];
    }

    cudaDeviceSynchronize();    

    exchangeBoundaryData<MPI_INT>(nTimeStep, mMPIBuffer); 

    for (auto & field : fieldChunks)
    {
        int threadsRequired = *std::max_element(mGeometrySize.begin(), mGeometrySize.end())*field.second;
        int dataAcquired = 0;
        int num_block_2d = (threadsRequired + blocksize_2d -1)/blocksize_2d;
        distributeData1D<T><<< num_block_2d, blocksize_2d >>>(field.first, field.second, dataAcquired, mMPIBuffer["recvL"].first, mMPIBuffer["recvR"].first,
                                                           nxLocal, nyLocal, nzLocal, haloWidth);
        dataAcquired += field.second*mGeometrySize[0];
    }

    cudaDeviceSynchronize();      
}

void MPIsetting1D::MPItransferData(int nTimeStep, std::vector<std::pair<float*, int>> fieldChunks, std::map<std::string, std::pair<float*, int> > mMPIBuffer)
{
    int blocksize_2d = 128;  // seems reduce the block size makes it a little faster, but around 128 is okay.
    for (auto & field : fieldChunks)
    {
        int threadsRequired = *std::max_element(mGeometrySize.begin(), mGeometrySize.end())*field.second;
        int dataAcquired = 0;
        int num_block_2d = (threadsRequired + blocksize_2d -1)/blocksize_2d;
        collectData1D<T><<< num_block_2d, blocksize_2d >>>(field.first, field.second, dataAcquired, mMPIBuffer["sendL"].first, mMPIBuffer["sendR"].first,
                                                        nxLocal, nyLocal, nzLocal, haloWidth);
        dataAcquired += field.second*mGeometrySize[0];
    }

    cudaDeviceSynchronize();    

    exchangeBoundaryData<MPI_FLOAT>(nTimeStep, mMPIBuffer); 

    for (auto & field : fieldChunks)
    {
        int threadsRequired = *std::max_element(mGeometrySize.begin(), mGeometrySize.end())*field.second;
        int dataAcquired = 0;
        int num_block_2d = (threadsRequired + blocksize_2d -1)/blocksize_2d;
        distributeData1D<T><<< num_block_2d, blocksize_2d >>>(field.first, field.second, dataAcquired, mMPIBuffer["recvL"].first, mMPIBuffer["recvR"].first,
                                                           nxLocal, nyLocal, nzLocal, haloWidth);
        dataAcquired += field.second*mGeometrySize[0];
    }

    cudaDeviceSynchronize();      
}

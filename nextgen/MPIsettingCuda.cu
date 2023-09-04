#include "MPIsetting.h"
#include "devicefunc.cu_inl"
#include <algorithm>

void MPIsetting::MPItransferData(int nTimeStep, std::vector<std::pair<float*, int>> fieldChunks)
{
    int blocksize_2d = 128;  // seems reduce the block size makes it a little faster, but around 128 is okay.
    for (auto & field : fieldChunks)
    {
        int threadsRequired = field.second*std::max_element(mGeometrySize.begin(), mGeometrySize.end());
        int dataAcquired = 0;
        int num_block_2d = (threadsRequired + blocksize_2d -1)/blocksize_2d;
        collectData1D<<< num_block_2d, blocksize_2d >>>(field.first, field.second, dataAcquired, 
                                                        nxLocal, nyLocal, nzLocal, haloWidth);
        dataAcquired += field.second*mGeometrySize[0];
    }

    cudaDeviceSynchronize();    

    exchangeBoundaryData(nTimeStep); 

    for (auto & field : fieldChunks)
    {
        int threadsRequired = field.second*std::max_element(mGeometrySize.begin(), mGeometrySize.end());
        int dataAcquired = 0;
        int num_block_2d = (threadsRequired + blocksize_2d -1)/blocksize_2d;
        distributeData1D<<< num_block_2d, blocksize_2d >>>(field.first, field.second, dataAcquired,
                                                           nxLocal, nyLocal, nzLocal, haloWidth);
        dataAcquired += field.second*mGeometrySize[0];
    }

    cudaDeviceSynchronize();      
}

__global__ void
collectData1D(float* field, int numFields, int offset, 
              int nxLocal, int nyLocal, int nzLocal, int haloWidth)
{
    int C = blockIdx.x * blockDim.x + threadIdx.x;
    int i, j, k, PF_id, fnx, fny, fnz;
    G2L_3D(C, i, j, k, PF_id, nxLocal, nyLocal, nzLocal);
    if ( (i<haloWidth) && (j<nyLocal) && (k<nzLocal) && (PF_id<numFields))
    {
        fnx = nxLocal + haloWidth*2;
        fny = nyLocal + haloWidth*2;
        fnz = nzLocal + haloWidth*2;
        // int field_indexL = i+hd+(j+hd)*fnx;
        int field_indexL = L2G_4D(i + haloWidth, j + haloWidth, k + haloWidth, PF_id, fnx, fny, fnz);
        int field_indexR = L2G_4D(i + nxLocal, j + haloWidth, k + haloWidth, PF_id, fnx, fny, fnz);

        mMPIBuffer["sendL"].first[C + offset] = field[field_indexL];
        mMPIBuffer["sendR"].first[C + offset] = field[field_indexR];
    }
}

__global__ void
distributeData1D(float* field, int numFields, int offset, 
                 int nxLocal, int nyLocal, int nzLocal, int haloWidth)
{
  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j, k, PF_id, fnx, fny, fnz;
  fnx = nxLocal + haloWidth*2;
  fny = nyLocal + haloWidth*2;
  fnz = nzLocal + haloWidth*2;

  G2L_3D(C, i, j, k, PF_id, nxLocal, nyLocal, nzLocal);

  if ( (i<haloWidth) && (j<nyLocal) && (k<nzLocal) && (PF_id<numFields))
  {

      // int field_indexL = i+hd+(j+hd)*fnx;
      int field_indexL = L2G_4D(i, j + haloWidth, k + haloWidth, PF_id, fnx, fny, fnz);
      int field_indexR = L2G_4D(i + nxLocal + haloWidth, j + haloWidth, k + haloWidth, PF_id, fnx, fny, fnz);

      field[field_indexL] = mMPIBuffer["recvL"].first[C + offset];
      field[field_indexR] = mMPIBuffer["recvR"].first[C + offset];
  }
}




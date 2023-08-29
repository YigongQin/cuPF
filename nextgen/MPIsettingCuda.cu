#include "MPIsetting.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "devicefunc.cu_inl"


void MPIsetting::MPItransferData(int nTimeStep, std::vector<float*, int> fieldChunks)
{
    int blocksize_2d = 128;  // seems reduce the block size makes it a little faster, but around 128 is okay.
    for (auto & field : fieldChunks)
    {
        int threadsRequired = field.second*std::max_element(mGeometrySize.begin(), mGeometrySize.end());
        int dataAcquired = 0;
        int num_block_2d = (threadsRequired + blocksize_2d -1)/blocksize_2d;
        collectData<<< num_block_2d, blocksize_2d >>>(field.first, dataAcquired);
        dataAcquired += field.second*mGeometrySize[0];
    }

    cudaDeviceSynchronize();    

    exchangeBoundaryData(nTimeStep); 

    for (auto & field : fieldChunks)
    {
        int threadsRequired = field.second*std::max_element(mGeometrySize.begin(), mGeometrySize.end());
        int dataAcquired = 0;
        int num_block_2d = (threadsRequired + blocksize_2d -1)/blocksize_2d;
        distributeData<<< num_block_2d, blocksize_2d >>>(field.first, dataAcquired);
        dataAcquired += field.second*mGeometrySize[0];
    }

    cudaDeviceSynchronize();      
}

__global__ void
MPIsetting1D::collectData(float* field, int offset)
{
    int C = blockIdx.x * blockDim.x + threadIdx.x;
    int hd = cP.haloWidth;
    int i, j, k, PF_id, fnx, fny, fnz;
    G2L_3D(C, i, j, k, PF_id, fnx, fny, fnz);
    
    int fid = C/(max_len*hd);
    int index = C-fid*max_len*hd;

    int i = index/hd;  // range [0,max]
    int j = index-i*hd;  // range [0,hd]
    int nx = fnx-2*hd; // actual size.
    int ny = fny-2*hd;

    if ( (i<ny) && (j<hd) && (fid<num_fields))
    {
        // left, right length ny
        int field_indexL = j+hd+(i+hd)*fnx;
        int field_indexR = j+nx+(i+hd)*fnx;

        mMPIBuffer["sendL"][index+fid*stridey] = field[offset + field_indexL];
        mMPIBuffer["sendR"][index+fid*stridey] = field[offset + field_indexR];
    }
}

__global__ void
MPIsetting1D::distributeData(float* field, int offset)
{
  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int hd = cP.haloWidth;
  int i, j, k, PF_id, fnx, fny, fnz;
  G2L_3D(C, i, j, k, PF_id, fnx, fny, fnz);

  int fid = C/(max_len*hd);
  int index = C-fid*max_len*hd;

  int i = index/hd;  // range [0,max]
  int j = index-i*hd;  // range [0,hd]
  int nx = fnx-2*hd; // actual size.
  int ny = fny-2*hd;

  if ( (i<ny) && (j<hd) && (fid<num_fields))
  {
      int field_indexL = j+(i+hd)*fnx;
      int field_indexR = j+nx+hd+(i+hd)*fnx;

      field[offset + field_indexL] = mMPIBuffer["sendL"][index+fid*stridey];
      field[offset + field_indexR] = mMPIBuffer["sendR"][index+fid*stridey];
  }
}




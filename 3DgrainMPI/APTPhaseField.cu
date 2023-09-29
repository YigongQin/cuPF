#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <mpi.h>
#include "CycleTimer.h"
#include "params.h"
#include "helper.h"
#include "PhaseField.h"
#include "APTPhaseField.h"
#include "devicefunc.cu_inl"
#include <iostream>

using namespace std;

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCKSIZE BLOCK_DIM_X*BLOCK_DIM_Y
#define HALO 1//halo in global region
#define REAL_DIM_X 14 //BLOCK_DIM_X-2*HALO
#define REAL_DIM_Y 14 //BLOCK_DIM_Y-2*HALO
#define LS -0.995
#define OMEGA 12
#define ZERO 0

#define TIPP 20
#define APT_NUM_PF 5
  


__constant__ GlobalConstants cP;

__inline__ __device__ float
kine_ani(float ux, float uy, float uz, float cosa, float sina, float cosb, float sinb){

   float a_s = 1.0f + 3.0f*cP.kin_delta;
   float epsilon = -4.0f*cP.kin_delta/a_s;
   float ux2 = cosa*cosb*ux  + sina*cosb*uy - sinb*uz;
         ux2 = ux2*ux2;
   float uy2 = -sina*ux      + cosa*uy      + 0;
         uy2 = uy2*uy2;      
   float uz2 = cosa*sinb*ux  + sina*sinb*uy + cosb*uz;
         uz2 = uz2*uz2;
   float MAG_sq = (ux2 + uy2 + uz2);
   float MAG_sq2= MAG_sq*MAG_sq;
   if (MAG_sq > cP.eps){
         return a_s*( 1.0f + epsilon*(ux2*ux2 + uy2*uy2 + uz2*uz2) / MAG_sq2);}
   else {return 1.0f;}
}


__global__ void
APTcollect_PF(float* PFs, float* phi, int* alpha_m, int* active_args){

   int C = blockIdx.x * blockDim.x + threadIdx.x;
   int length = cP.length;
   if (C<length){
     int argmax = 0;
   // for loop to find the argmax of the phase field
     for (int PF_id=0; PF_id<cP.NUM_PF; PF_id++){
       int loc = C + length*PF_id; 
       int max_loc = C + length*argmax;
       if (PFs[loc]>PFs[max_loc]) {argmax=PF_id;}
     }
    
   int max_loc_f = C + length*argmax; 
   if (PFs[max_loc_f]>LS){
      phi[C] = PFs[max_loc_f]; 
      alpha_m[C] = active_args[max_loc_f];
    }
   }
}


__global__ void
APTini_PF(float* PFs, float* phi, int* alpha_m, int* active_args){

   int C = blockIdx.x * blockDim.x + threadIdx.x;
  if ( C<cP.length ) {
    if ( phi[C]>LS){
      PFs[C] = phi[C];
      active_args[C] = alpha_m[C];
    }

  }
}



__global__ void
APTsetBC3D(float* ph, int* active_args, int max_area,
                   int processorIDX, int processorIDY, int processorIDZ, int numProcessorX, int numProcessorY, int numProcessorZ)
{

   // dimension with R^{2D} * PF

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int fnx = cP.fnx, fny = cP.fny, fnz = cP.fnz, NUM_PF = cP.NUM_PF;

  int pf = index/max_area;
  int bc_idx = index- pf*max_area;     

  int area_z = fnx*fny;
  if ( (bc_idx<area_z) && (pf<NUM_PF))
  {
     int zj = bc_idx/fnx;
     int zi = bc_idx - zj*fnx;
     int exchangeZIndex_d = cP.bcZ==0 ? 2 : fnz-2;
     int exchangeZIndex_u = cP.bcZ==0 ? fnz-3 : 1;
     if (processorIDZ==0)
     {
        int d_out = L2G_4D(zi, zj, 0, pf, fnx, fny, fnz);
        int d_in  = L2G_4D(zi, zj, exchangeZIndex_d, pf, fnx, fny, fnz);
        ph[d_out] = ph[d_in];
        active_args[d_out] = active_args[d_in];
     }
     if (processorIDZ==numProcessorZ-1)
     {
        int u_out = L2G_4D(zi, zj, fnz-1, pf, fnx, fny, fnz);
        int u_in  = L2G_4D(zi, zj, exchangeZIndex_u, pf, fnx, fny, fnz);
        ph[u_out] = ph[u_in];
        active_args[u_out] = active_args[u_in];
     }
  }

  int area_y = fnx*fnz;
  if ( (bc_idx<area_y) && (pf<NUM_PF))
  {
     int zk = bc_idx/fnx;
     int zi = bc_idx - zk*fnx;
     int exchangeYIndex_b = cP.bcY==0 ? 2 : fny-2;
     int exchangeYIndex_t = cP.bcY==0 ? fny-3 : 1;
     if (processorIDY==0)
     {
        int b_out = L2G_4D(zi, 0, zk, pf, fnx, fny, fnz);
        int b_in = L2G_4D(zi, exchangeYIndex_b, zk, pf, fnx, fny, fnz);
        ph[b_out] = ph[b_in];
        active_args[b_out] = active_args[b_in];
     }
     if (processorIDY==numProcessorY-1)
     {
        int t_out = L2G_4D(zi, fny-1, zk, pf, fnx, fny, fnz);
        int t_in = L2G_4D(zi, exchangeYIndex_t, zk, pf, fnx, fny, fnz);
        ph[t_out] = ph[t_in];
        active_args[t_out] = active_args[t_in];
     }
  }

  int area_x = fny*fnz;
  if ( (bc_idx<area_x) && (pf<NUM_PF))
  {
     int zk = bc_idx/fny;
     int zj = bc_idx - zk*fny;
     int exchangeXIndex_l = cP.bcX==0 ? 2 : fnx-2;
     int exchangeXIndex_r = cP.bcX==0 ? fnx-3 : 1;
     if (processorIDX==0)
     {
        int l_out = L2G_4D(0, zj, zk, pf, fnx, fny, fnz);
        int l_in = L2G_4D(exchangeXIndex_l, zj, zk, pf, fnx, fny, fnz);
        ph[l_out] = ph[l_in];
        active_args[l_out] = active_args[l_in];
     }
     if (processorIDX==numProcessorX-1)
     {
        int r_out = L2G_4D(fnx-1, zj, zk, pf, fnx, fny, fnz);
        int r_in = L2G_4D(exchangeXIndex_r, zj, zk, pf, fnx, fny, fnz);
        ph[r_out] = ph[r_in];
        active_args[r_out] = active_args[r_in]; 
     }
  }
}


// psi equation
__global__ void
APTrhs_psi(float t, float* x, float* y, float* z, float* ph, float* ph_new, int* aarg, int* aarg_new, ThermalInputData thm)
{

  int C = blockIdx.x * blockDim.x + threadIdx.x; 
  int i, j, k, PF_id;
  int fnx = cP.fnx, fny = cP.fny, fnz = cP.fnz, NUM_PF = cP.NUM_PF, length = cP.length;
  G2L_3D(C, i, j, k, PF_id, fnx, fny, fnz);

  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) && (k>0) && (k<fnz-1) ) 
  {

  //=============== load active phs ================
        int local_args[APT_NUM_PF]; // local active PF indices
        float local_phs[7][APT_NUM_PF]; // active ph for each thread ,use 7-point stencil


        int globalC, target_index, stencil, arg_index;
        for (int arg_index = 0; arg_index<NUM_PF; arg_index++)
        {
            local_args[arg_index] = -1;
            for (stencil=0; stencil<7; stencil++)
            {
                local_phs[stencil][arg_index] = -1.0f;
            } 
        }
        stencil = 0;
        for (int zi = -1; zi<=1; zi++)
        {
            for (int yi = -1; yi<=1; yi++){
                for (int xi = -1; xi <=1; xi++){
                    for (int pf_id = 0; pf_id<NUM_PF; pf_id++){
                        globalC = C + xi + yi*fnx + zi*fnx*fny + pf_id*fnx*fny*fnz;
                        target_index = aarg[globalC];
                        if (target_index==-1){
                            continue;
                        }
                        // otherwise find/update the local_args and local_phs
                        for (arg_index = 0; arg_index<NUM_PF; arg_index++){
                            if (local_args[arg_index]==target_index) {
                                break;
                            }
                            if (local_args[arg_index] == -1){
                                local_args[arg_index] = target_index; 
                                break;
                            }
                        }
                        if ( abs(xi) + abs(yi) + abs(zi) <=1) {
                            local_phs[stencil][arg_index] = ph[globalC];
                        }
                    }
                    if ( abs(xi) + abs(yi) + abs(zi) <=1) {stencil += 1;}
                }
            }
        }

       float Dt = thm.t_mac[1] - thm.t_mac[0];
       float Dx = thm.X_mac[1] - thm.X_mac[0]; 

       for (arg_index = 0; arg_index<NUM_PF; arg_index++)
       {
            if (local_args[arg_index]==-1)
            {
                aarg_new[C+arg_index*length] = -1;
                ph_new[C+arg_index*length] = -1.0f;
            }
            else
            {

                // start dealing with one specific PF

                PF_id = local_args[arg_index]; // global PF index to find the right orientation
                float phD=local_phs[0][arg_index], phB=local_phs[1][arg_index], phL=local_phs[2][arg_index], \
                phC=local_phs[3][arg_index], phR=local_phs[4][arg_index], phT=local_phs[5][arg_index], phU=local_phs[6][arg_index];

                float phxn = ( phR - phL ) * 0.5f;
                float phyn = ( phT - phB ) * 0.5f;
                float phzn = ( phU - phD ) * 0.5f;

                float cosa, sina, cosb, sinb;
                if (phC>LS)
                {
                        int theta_id = PF_id % cP.num_theta;
                        sina = thm.sint[theta_id];
                        cosa = thm.cost[theta_id];
                        sinb = thm.sint[theta_id+cP.num_theta];
                        cosb = thm.cost[theta_id+cP.num_theta];
                }
                else
                {
                        sina = 0.0f;
                        cosa = 1.0f;
                        sinb = 0.0f;
                        cosb = 1.0f;
                }

                float A2 = kine_ani(phxn,phyn,phzn,cosa,sina,cosb,sinb);

                float diff =  phR + phL + phT + phB + phU + phD - 6*phC;
                float Tinterp;

                if (cP.thermalType == 1)
                {
                    Tinterp = cP.G*(z[k] - cP.z0) - cP.underCoolingRate*1e6 *t;   
                }
                else
                {
                    Tinterp = interp4Dtemperature(thm.T_3D, x[i] - thm.X_mac[0], y[j] - thm.Y_mac[0], z[k] - thm.Z_mac[0], t - thm.t_mac[0], 
                        cP.Nx, cP.Ny, cP.Nz, cP.Nt, Dx, Dt);
                }
                
                float Up = Tinterp/(cP.L_cp);  
                float repul=0.0f;
                for (int pf_id=0; pf_id<NUM_PF; pf_id++)
                {
                if (pf_id!=arg_index) 
                {
                    repul += 0.25f*(local_phs[3][pf_id]+1.0f)*(local_phs[3][pf_id]+1.0f);
                }
                }

                float rhs_psi = diff * cP.hi*cP.hi + (1.0f-phC*phC)*phC \
                    - cP.lamd*Up* ( (1.0f-phC*phC)*(1.0f-phC*phC) - 0.5f*OMEGA*(phC+1.0f)*repul);
                float dphi = rhs_psi / A2; 
                ph_new[C+arg_index*length] = phC  +  cP.dt * dphi; 
                if (phC  +  cP.dt * dphi <-0.9999)
                {
                    aarg_new[C+arg_index*length] = -1;
                }
                else
                {
                    aarg_new[C+arg_index*length] = local_args[arg_index];
                }

            }
        }
     }
} 


__global__ void 
APTmove_frame(float* ph_buff, int* arg_buff, float* z_buff, float* ph, int* arg, float* z, int* alpha, int* alpha_full, int* loss_area, int move_count){

    int C = blockIdx.x * blockDim.x + threadIdx.x;
    int i, j, k, PF_id;
    int fnx = cP.fnx, fny = cP.fny, fnz = cP.fnz, NUM_PF = cP.NUM_PF;
    G2L_4D(C, i, j, k, PF_id, fnx, fny, fnz);

    if ( (i==0) && (j==0) && (k>0) && (k<fnz-2) && (PF_id==0) ) {
        z_buff[k] = z[k+1];}

    if ( (i==0) && (j==0) &&  (k==fnz-2) && (PF_id==0) ) {        
        z_buff[k] = 2*z[fnz-2] - z[fnz-3];}

    if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) && (k>0) && (k<fnz-2) && (PF_id<NUM_PF) ) {     
        ph_buff[C] = ph[C+fnx*fny];
        arg_buff[C] = arg[C+fnx*fny];
    }

    if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) && (k==fnz-2) && (PF_id<NUM_PF) ) {

        ph_buff[C] = -1.0f;
        arg_buff[C] = -1;
    }

    // add last layer of alpha to alpha_full[move_count]
    if ( (i<fnx) && (j<fny) && (k==1) && (PF_id==0) ) {

        alpha_full[move_count*fnx*fny+C] = alpha[C];
        atomicAdd(loss_area+alpha[C]-1,1);
        //printf("%d ", alpha[C]);
    }


}

__global__ void
APTcopy_frame(float* ph_buff, int* arg_buff, float* z_buff, float* ph, int* arg, float* z){

  int C = blockIdx.x * blockDim.x + threadIdx.x; 
  int i, j, k, PF_id;
  int fnx = cP.fnx, fny = cP.fny, fnz = cP.fnz, NUM_PF = cP.NUM_PF;
  G2L_4D(C, i, j, k, PF_id, fnx, fny, fnz);

  if ( (i == 0) && (j==0) && (k>0) && (k < fnz-1) && (PF_id==0) ){
        z[k] = z_buff[k];}

  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) && (k>0) && (k<fnz-1) && (PF_id<NUM_PF) ) { 
        ph[C] = ph_buff[C];
        arg[C] = arg_buff[C];
    }


}


__global__ void
init_rand_num(curandState *state, int seed_val, int len_plus)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id<len_plus) curand_init(seed_val, id, 0, &state[id]);
}

__global__ void
init_nucl_status(float* ph, int* nucl_status, int cnx, int cny, int cnz, int fnx, int fny)
{
  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j, k, PF_id;
  G2L_3D(C, i, j, k, PF_id, cnx, cny, cnz);

  if ( (i<cnx) && (j<cny) && (k<cnz) ) 
  {
      int glob_i = (2*cP.pts_cell+1)*i + cP.pts_cell;
      int glob_j = (2*cP.pts_cell+1)*j + cP.pts_cell;
      int glob_k = (2*cP.pts_cell+1)*k + cP.pts_cell;

      int glob_C = glob_k*fnx*fny + glob_j*fnx + glob_i;   

      if (ph[glob_C]>LS)
      {
          nucl_status[C] = 1;
      } 
      else 
      {
        nucl_status[C] = 0;
      }
  }
}

__inline__ __device__ float 
nuncl_possibility(float delT, float d_delT, float nuc_Nmax)
{
  float slope = -0.5f*(delT-cP.undcool_mean)*(delT-cP.undcool_mean)/cP.undcool_std/cP.undcool_std;
  slope = expf(slope); 
  float density = nuc_Nmax/(sqrtf(2.0f*M_PI)*cP.undcool_std) *slope*d_delT;
   // float nuc_posb = 4.0f*cP.nuc_rad*cP.nuc_rad*density; // 2D
  float nuc_posb = 8.0f*cP.nuc_rad*cP.nuc_rad*cP.nuc_rad*density; // 3D
  return nuc_posb; 
}



__global__ void
add_nucl(float* ph, int* arg, int* nucl_status, int cnx, int cny, int cnz, float* x, float* y, float* z, int fnx, int fny, int fnz, curandState* states, 
        float dt, float t, ThermalInputData thm, bool useInitialUnderCooling)
{
  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j, k, PF_id;
  G2L_3D(C, i, j, k, PF_id, cnx, cny, cnz);

  float Dt = thm.t_mac[1]-thm.t_mac[0];
  float Dx = thm.X_mac[1]-thm.X_mac[0]; 

  if ( (i<cnx) && (j<cny) && (k<cnz)) 
  {
    if (nucl_status[C]==0)
    {
      int glob_i = (2*cP.pts_cell+1)*i + cP.pts_cell + cP.haloWidth;;
      int glob_j = (2*cP.pts_cell+1)*j + cP.pts_cell + cP.haloWidth;; 
      int glob_k = (2*cP.pts_cell+1)*k + cP.pts_cell + cP.haloWidth;; 
      int glob_C = glob_k*fnx*fny + glob_j*fnx + glob_i;
    
      for (int pf_id = 0; pf_id<cP.NUM_PF; pf_id++)
      {
        int loc = glob_C + pf_id*fnx*fny*fnz;
        if (ph[loc]>LS)
        {
          nucl_status[C] = 1;
        }
      }

      if (nucl_status[C]==0)
      {
        float nuc_posb;
        if (useInitialUnderCooling)
        {
            float delT = cP.underCoolingRate0*1e6*(t+dt);
            float d_delT = cP.underCoolingRate0*1e6*dt;
            nuc_posb = nuncl_possibility(delT, d_delT, cP.nuc_Nmax0);
        }
        else
        {
            float T_cell = interp4Dtemperature(thm.T_3D, x[glob_i] - thm.X_mac[0], y[glob_j] - thm.Y_mac[0], z[glob_k] - thm.Z_mac[0], t - thm.t_mac[0], 
                cP.Nx, cP.Ny, cP.Nz, cP.Nt, Dx, Dt);
            float T_cell_dt = interp4Dtemperature(thm.T_3D, x[glob_i] - thm.X_mac[0], y[glob_j] - thm.Y_mac[0], z[glob_k] - thm.Z_mac[0], t+dt - thm.t_mac[0], 
                cP.Nx, cP.Ny, cP.Nz, cP.Nt, Dx, Dt);                                        
            float delT = - T_cell_dt;
            float d_delT = T_cell - T_cell_dt;
            nuc_posb = nuncl_possibility(delT, d_delT, cP.nuc_Nmax);
        }
        
        //printf("nucleation possibility at cell no. %f, %f \n", delT, d_delT);
        if (curand_uniform(states+C)<nuc_posb)
        {
            int rand_PF = curand(states+C);
            rand_PF = rand_PF>0 ? rand_PF : -rand_PF;
            printf("time %f, nucleation starts at location %f, %f, %f, get the same orientation with grain no. %d\n", 1000000.0f*t, x[glob_i], y[glob_j], z[glob_k], rand_PF);

            for (int lock=-cP.pts_cell; lock<=cP.pts_cell; lock++)
            {
                for (int locj=-cP.pts_cell; locj<=cP.pts_cell; locj++)
                {
                    for (int loci=-cP.pts_cell; loci<=cP.pts_cell; loci++)
                    {
                        int os_loc = glob_C + lock*fnx*fny + locj*fnx + loci;
                        float dist_C = cP.dx*( (1.0f+cP.pts_cell)/2.0f - sqrtf(loci*loci + locj*locj + lock*lock) );
                        ph[os_loc] = tanhf( dist_C /cP.sqrt2 );
                        if (ph[os_loc]>-0.9999f)
                        {
                            arg[os_loc] = rand_PF;
                        }
                    }
                }
            }
            nucl_status[C] = 1;
        } 
      }
    }
  }
}



APTPhaseField::~APTPhaseField() {
    if (x){
        delete [] x;
        delete [] y;
        delete [] z;
    }

    if (phi){
        delete [] phi;
        delete [] psi;
        delete [] alpha;
    }

    if (x_device){
        cudaFree(x_device); cudaFree(y_device); 
        cudaFree(z_device); cudaFree(z_device2);
    }
    if (phi_new){
          cudaFree(phi_old); cudaFree(phi_new);
  //cudaFree(nucl_status); 
    // cudaFree(dStates); //cudaFree(random_nums);
     cudaFree(PFs_new); cudaFree(PFs_old);
     cudaFree(alpha_m); //cudaFree(argmax);
     cudaFree(d_alpha_full);
    }

}


void APTPhaseField::cudaSetup() 
{
    MPIsetting* mpiManager = GetMPIManager();

    int num_gpus_per_node = 4;
    int device_id_innode = mpiManager->rank % num_gpus_per_node;
    //gpu_name = cuda.select_device( )
    cudaSetDevice(device_id_innode); 

    printCudaInfo(mpiManager->rank,device_id_innode);

    params.NUM_PF = APT_NUM_PF;
    NUM_PF = params.NUM_PF;
    printf("number of PFs %d \n", NUM_PF); 

    // allocate device memory and copy the data
    cudaMalloc((void **)&x_device, sizeof(float) * fnx);
    cudaMalloc((void **)&y_device, sizeof(float) * fny);
    cudaMalloc((void **)&z_device, sizeof(float) * fnz);
    cudaMalloc((void **)&z_device2, sizeof(float) * fnz);

    cudaMalloc((void **)&phi_old,  sizeof(float) * length);
    cudaMalloc((void **)&phi_new,  sizeof(float) * length);
    cudaMalloc((void **)&alpha_m,    sizeof(int) * length);  
    cudaMalloc((void **)&d_alpha_full,   sizeof(int) * fnx*fny*fnz_f);  

    cudaMalloc((void **)&PFs_old,    sizeof(float) * length * NUM_PF);
    cudaMalloc((void **)&PFs_new,    sizeof(float) * length * NUM_PF);
    cudaMalloc((void **)&active_args_old,    sizeof(int) * length * NUM_PF);
    cudaMalloc((void **)&active_args_new,    sizeof(int) * length * NUM_PF);
    cudaMemset(active_args_old,-1,sizeof(int) * length * NUM_PF);
    cudaMemset(active_args_new,-1,sizeof(int) * length * NUM_PF);
    args_cpu = new int[length * NUM_PF];

    cudaMemcpy(x_device, x, sizeof(float) * fnx, cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y, sizeof(float) * fny, cudaMemcpyHostToDevice);
    cudaMemcpy(z_device, z, sizeof(float) * fnz, cudaMemcpyHostToDevice);
    cudaMemcpy(z_device2, z, sizeof(float) * fnz, cudaMemcpyHostToDevice);
    cudaMemcpy(phi_old, phi, sizeof(float) * length, cudaMemcpyHostToDevice);
    cudaMemcpy(phi_new, phi, sizeof(float) * length, cudaMemcpyHostToDevice);

    cudaMemcpy(alpha_m, alpha, sizeof(int) * length, cudaMemcpyHostToDevice);

    // pass all the read-only params into global constant
    cudaMemcpyToSymbol(cP, &params, sizeof(GlobalConstants) );

    // create forcing field
    //cudaMalloc((void **)&(Mgpu),  sizeof(ThermalInputData)); 
    cudaMalloc((void **)&(Mgpu.X_mac),  sizeof(float) * mac.Nx);
    cudaMalloc((void **)&(Mgpu.Y_mac),  sizeof(float) * mac.Ny);
    cudaMalloc((void **)&(Mgpu.Z_mac),  sizeof(float) * mac.Nz);
    cudaMalloc((void **)&(Mgpu.t_mac),    sizeof(float) * mac.Nt);
    cudaMalloc((void **)&(Mgpu.T_3D),    sizeof(float) * mac.Nx*mac.Ny*mac.Nz*mac.Nt);
    cudaMalloc((void **)&(Mgpu.theta_arr),    sizeof(float) * (2*params.num_theta+1) );
    cudaMalloc((void **)&(Mgpu.cost),    sizeof(float) * (2*params.num_theta+1) );
    cudaMalloc((void **)&(Mgpu.sint),    sizeof(float) * (2*params.num_theta+1) );
    cudaMemcpy(Mgpu.X_mac, mac.X_mac, sizeof(float) * mac.Nx, cudaMemcpyHostToDevice);  
    cudaMemcpy(Mgpu.Y_mac, mac.Y_mac, sizeof(float) * mac.Ny, cudaMemcpyHostToDevice); 
    cudaMemcpy(Mgpu.Z_mac, mac.Z_mac, sizeof(float) * mac.Nz, cudaMemcpyHostToDevice);  
    cudaMemcpy(Mgpu.t_mac, mac.t_mac, sizeof(float) * mac.Nt, cudaMemcpyHostToDevice);  
    cudaMemcpy(Mgpu.T_3D, mac.T_3D, sizeof(float) * mac.Nt* mac.Nx* mac.Ny* mac.Nz, cudaMemcpyHostToDevice);   
    cudaMemcpy(Mgpu.theta_arr, mac.theta_arr, sizeof(float) * (2*params.num_theta+1), cudaMemcpyHostToDevice);
    cudaMemcpy(Mgpu.cost, mac.cost, sizeof(float) * (2*params.num_theta+1), cudaMemcpyHostToDevice);
    cudaMemcpy(Mgpu.sint, mac.sint, sizeof(float) * (2*params.num_theta+1), cudaMemcpyHostToDevice);
    

}

void MovingDomain::allocateMovingDomain(int numGrains, int MovingDirectoinSize)
{
    tip_thres = (int) (0.8*MovingDirectoinSize);
    printf("max tip can go: %d\n", tip_thres); 

    meanx_host = new float[MovingDirectoinSize];
    loss_area_host = new int[numGrains];
    memset(loss_area_host, 0, sizeof(int) * numGrains);

    cudaMalloc((void **)&meanx_device, sizeof(float) * MovingDirectoinSize);
    cudaMemset(meanx_device,0, sizeof(float) * MovingDirectoinSize);

    cudaMalloc((void **)&loss_area_device, sizeof(int) * numGrains); 
    cudaMemset(loss_area_device, 0, sizeof(int) * numGrains);
}

void APTPhaseField::getLineQoIs(MovingDomain* movingDomainManager)
{
    float locationInMovingDomain = (movingDomainManager->move_count + movingDomainManager->lowsl -1)*params.W0*params.dx; 
    float threshold = params.z0 + params.top*(movingDomainManager->samples+1)/params.nts;

    if (locationInMovingDomain > threshold) 
    {
        printf("threshold %f \n", threshold);
        cudaMemset(alpha_m, 0, sizeof(int) * length);
        APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
        cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost); 
        cudaMemcpy(args_cpu, active_args_old, NUM_PF*length * sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(movingDomainManager->loss_area_host, movingDomainManager->loss_area_device, params.num_theta * sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(z, z_device, fnz * sizeof(int),cudaMemcpyDeviceToHost); 
        //QoIs based on alpha field
        movingDomainManager->cur_tip=0;
        qois->calculateLineQoIs(params, movingDomainManager->cur_tip, alpha, movingDomainManager->samples+1, z,
                                movingDomainManager->loss_area_host, movingDomainManager->move_count);
        movingDomainManager->samples += 1;
    }
}

void APTPhaseField::moveDomain(MovingDomain* movingDomainManager)
{
    tip_mvf(&movingDomainManager->tip_front, PFs_old, movingDomainManager->meanx_device, movingDomainManager->meanx_host, fnx, fny, fnz, NUM_PF);
    while (movingDomainManager->tip_front >= movingDomainManager->tip_thres)
    {
       APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
       APTmove_frame<<< num_block_PF, blocksize_2d >>>(PFs_new, active_args_new, z_device2, PFs_old, active_args_old, z_device, alpha_m, d_alpha_full, 
                                                       movingDomainManager->loss_area_device, movingDomainManager->move_count);
       APTcopy_frame<<< num_block_PF, blocksize_2d >>>(PFs_new, active_args_new, z_device2, PFs_old, active_args_old, z_device);
       movingDomainManager->move_count += 1;
       movingDomainManager->tip_front -=1 ;
    }

    APTsetBC3D<<<num_block_PF1d, blocksize_1d>>>(PFs_old, active_args_old, max_area,
                                                 0, 0, 0, 1, 1, 1);
    movingDomainManager->lowsl = 1;
    APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
    cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
    qois->sampleHeights(movingDomainManager->lowsl, alpha, fnx, fny, fnz);
}


void APTPhaseField::setBC(bool useLineConfig, float* ph, int* active_args)
{
    MPIsetting* mpiManager = GetMPIManager();

    APTsetBC3D<<<num_block_PF1d, blocksize_1d>>>(ph, active_args, max_area,
             mpiManager->processorIDX, mpiManager->processorIDY, mpiManager->processorIDZ, mpiManager->numProcessorX, mpiManager->numProcessorY, mpiManager->numProcessorZ);

}


void APTPhaseField::evolve()
{
    const DesignSettingData* designSetting = GetSetDesignSetting(); 
    MPIsetting1D* mpiManager = dynamic_cast<MPIsetting1D*> (GetMPIManager());
    MovingDomain* movingDomainManager = new MovingDomain();

    blocksize_1d = 128;
    blocksize_2d = 128;
    num_block_2d = (length+blocksize_2d-1)/blocksize_2d;
    num_block_PF = (length*NUM_PF+blocksize_2d-1)/blocksize_2d;
    max_area = max(fnz*fny,max(fny*fnx,fnx*fnz));
    num_block_PF1d =  ( max_area*NUM_PF +blocksize_1d-1)/blocksize_1d;
    printf("block size %d, # blocks %d\n", blocksize_2d, num_block_2d); 


    // initial condition
    set_minus1<<< num_block_PF, blocksize_2d>>>(PFs_old,length*NUM_PF);
    set_minus1<<< num_block_PF, blocksize_2d>>>(PFs_new,length*NUM_PF);
    APTini_PF<<< num_block_PF, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
    APTini_PF<<< num_block_PF, blocksize_2d >>>(PFs_new, phi_old, alpha_m, active_args_new);
    
    curandState* dStates;
    if (designSetting->includeNucleation)
    {
      cnx = (fnx - 1 - 2*params.haloWidth)/(2*params.pts_cell+1);
      cny = (fny - 1 - 2*params.haloWidth)/(2*params.pts_cell+1);
      cnz = (fnz - 1 - 2*params.haloWidth)/(2*params.pts_cell+1);
      num_block_c = (cnx*cny*cnz + blocksize_2d-1)/blocksize_2d;   
      int lenCell = cnx*cny*cnz;

      cudaMalloc((void **) &dStates, sizeof(curandState) * (lenCell+params.noi_period));
      cudaMalloc((void **) &nucleationStatus, sizeof(int) * lenCell);
      init_rand_num<<<(lenCell+params.noi_period)/blocksize_2d, blocksize_2d>>>(dStates, params.seed_val, lenCell+params.noi_period);
      init_nucl_status<<<num_block_c, blocksize_2d>>>(phi_old, nucleationStatus, cnx, cny, cnz, fnx, fny);
    }

    if (mpiManager->numProcessor >1)
    {
        mpiManager->PFBuffer = mpiManager->createBoundaryBuffer<float>(NUM_PF);
        mpiManager->ArgBuffer = mpiManager->createBoundaryBuffer<int>(NUM_PF);

        mpiManager->data_new_float.push_back(std::make_pair(PFs_new, NUM_PF));
        mpiManager->data_old_float.push_back(std::make_pair(PFs_old, NUM_PF));
        mpiManager->data_new_int.push_back(std::make_pair(active_args_new, NUM_PF));
        mpiManager->data_old_int.push_back(std::make_pair(active_args_old, NUM_PF));

        for (auto & buffer : mpiManager->PFBuffer)
        {
            cudaMalloc((void **)&(buffer.second.first),  sizeof(float)*buffer.second.second );
        }
        for (auto & buffer : mpiManager->ArgBuffer)
        {
            cudaMalloc((void **)&(buffer.second.first),  sizeof(int)*buffer.second.second );
        }     
        mpiManager->MPItransferData(0, mpiManager->data_new_float, mpiManager->PFBuffer);
        mpiManager->MPItransferData(1, mpiManager->data_old_float, mpiManager->PFBuffer);
        mpiManager->MPItransferData(2, mpiManager->data_new_int, mpiManager->ArgBuffer);
        mpiManager->MPItransferData(3, mpiManager->data_old_int, mpiManager->ArgBuffer);
    }

    setBC(designSetting->useLineConfig, PFs_old, active_args_old);
    setBC(designSetting->useLineConfig, PFs_new, active_args_new);

    // get initial fields
    APTrhs_psi<<< num_block_2d, blocksize_2d >>>(0, x_device, y_device, z_device, PFs_old, PFs_new, active_args_old, active_args_new, Mgpu);

    float t_cur_step;
    int kt = 0;
    if (params.preMt>0)
    {
        for (kt=0; kt<params.preMt/2; kt++)
        {
            if (mpiManager->numProcessor >1 && mpiManager->haloWidth == 1)
            {
                mpiManager->MPItransferData(4*kt + 10, mpiManager->data_new_float, mpiManager->PFBuffer);
                mpiManager->MPItransferData(4*kt + 12, mpiManager->data_new_int, mpiManager->ArgBuffer);
            }
            setBC(designSetting->useLineConfig, PFs_new, active_args_new);
    
            t_cur_step = (2*kt+1)*params.dt*params.tau0;
            APTrhs_psi<<< num_block_2d, blocksize_2d >>>(t_cur_step, x_device, y_device, z_device, PFs_new, PFs_old, active_args_new, active_args_old, Mgpu);

            add_nucl<<<num_block_c, blocksize_2d>>>(PFs_old, active_args_old, nucleationStatus, cnx, cny, cnz, x_device, y_device, z_device, fnx, fny, fnz, dStates, \
                    2.0f*params.dt*params.tau0, t_cur_step, Mgpu, true); 

            if (mpiManager->numProcessor >1 && ((2*kt + 2)%mpiManager->haloWidth)==0 )
            {
                mpiManager->MPItransferData(4*kt + 11, mpiManager->data_old_float, mpiManager->PFBuffer);
                mpiManager->MPItransferData(4*kt + 13, mpiManager->data_old_int, mpiManager->ArgBuffer);
            }
            setBC(designSetting->useLineConfig, PFs_old, active_args_old);
    
            t_cur_step = (2*kt+2)*params.dt*params.tau0;
            APTrhs_psi<<< num_block_2d, blocksize_2d >>>(t_cur_step, x_device, y_device, z_device, PFs_old, PFs_new, active_args_old, active_args_new, Mgpu);
        }
        
    }
    cudaMemset(alpha_m, 0, sizeof(int) * length);
    APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_new, phi_old, alpha_m, active_args_new);
    cudaMemcpy(alpha, alpha_m, length * sizeof(int), cudaMemcpyDeviceToHost);
    
    APTini_PF<<< num_block_PF, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
    APTini_PF<<< num_block_PF, blocksize_2d >>>(PFs_new, phi_old, alpha_m, active_args_new);
    
    if (designSetting->useLineConfig)
    {
        movingDomainManager->allocateMovingDomain(params.num_theta, fnz);
        qois->calculateLineQoIs(params, movingDomainManager->cur_tip, alpha, 0, z, movingDomainManager->loss_area_host, movingDomainManager->move_count);
    }
    else
    {
        if (designSetting->includeNucleation == false)
        {
            qois->calculateQoIs(params, alpha, 0);
        }
    }

    int qoikts = params.Mt/params.nts;
    int fieldkts = designSetting->save3DField>0 ? params.Mt/designSetting->save3DField : 1e8;
    printf("steps between qois %d, no. qois %d\n", qoikts, params.nts);
    printf("steps between fields %d, no. fields %d\n", fieldkts, designSetting->save3DField);

    int numComm = 0;
    kt = 0;

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();

    for (kt=0; kt<params.Mt/2; kt++)
    {
        //for (int kt=0; kt<0; kt++){
        if (mpiManager->numProcessor >1 && mpiManager->haloWidth == 1)
        {
            mpiManager->MPItransferData(4*kt + 10, mpiManager->data_new_float, mpiManager->PFBuffer);
            mpiManager->MPItransferData(4*kt + 12, mpiManager->data_new_int, mpiManager->ArgBuffer);
            numComm++;
        }
        setBC(designSetting->useLineConfig, PFs_new, active_args_new);

        t_cur_step = (2*kt+1)*params.dt*params.tau0;
        APTrhs_psi<<< num_block_2d, blocksize_2d >>>(t_cur_step, x_device, y_device, z_device, PFs_new, PFs_old, active_args_new, active_args_old, Mgpu);

        if (designSetting->includeNucleation)
        {
            add_nucl<<<num_block_c, blocksize_2d>>>(PFs_old, active_args_old, nucleationStatus, cnx, cny, cnz, x_device, y_device, z_device, fnx, fny, fnz, dStates, \
                2.0f*params.dt*params.tau0, t_cur_step, Mgpu, false); 
        }


        if (mpiManager->numProcessor >1 && ((2*kt + 2)%mpiManager->haloWidth)==0 )
        {
            mpiManager->MPItransferData(4*kt + 11, mpiManager->data_old_float, mpiManager->PFBuffer);
            mpiManager->MPItransferData(4*kt + 13, mpiManager->data_old_int, mpiManager->ArgBuffer);
            numComm++;
        }

        setBC(designSetting->useLineConfig, PFs_old, active_args_old);

        if (designSetting->useLineConfig)
        {
            getLineQoIs(movingDomainManager);
            if (movingDomainManager->samples==params.nts)
            {
                printf("sample all %d heights\n", params.nts);
                break;
            }
            if ( (2*kt+2)%TIPP==0) 
            {
                moveDomain(movingDomainManager);
            }
        }
        else
        {
            if ((2*kt+2)%qoikts==0)
            {
                APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
                cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost); 
                qois->calculateQoIs(params, alpha, (2*kt+2)/qoikts);
            }
        }

        if ((2*kt+2)%fieldkts==0)
        {
            APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
            cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost); 
            if (designSetting->useLineConfig)
            {
                cudaMemcpy(alpha_i_full, d_alpha_full, fnx*fny*fnz_f * sizeof(int),cudaMemcpyDeviceToHost);
                cudaMemcpy(alpha_i_full+movingDomainManager->move_count*fnx*fny, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);        
            }           
            OutputField(2*kt+2);
        }

        t_cur_step = (2*kt+2)*params.dt*params.tau0;
        APTrhs_psi<<< num_block_2d, blocksize_2d >>>(t_cur_step, x_device, y_device, z_device, PFs_old, PFs_new, active_args_old, active_args_new, Mgpu);
        //kt++;
   }


   cudaDeviceSynchronize();
   double endTime = CycleTimer::currentSeconds();
   printf("time for %d iterations: %f s\n", 2*kt, endTime-startTime);
   printf("no. communications performed %d \n", numComm);
   params.Mt = 2*kt; // the actual no. time steps
   
  // cudaMemset(alpha_m, 0, sizeof(int) * length);
  // APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old); 
  // cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);


}









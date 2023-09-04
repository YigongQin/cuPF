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
__constant__ MPIsetting cM;

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
APTset_BC_3D(float* ph, int* active_args, int max_area){
   // periodic x-y plance, no flux in z 
   // dimension with R^{2D} * PF

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int fnx = cP.fnx, fny = cP.fny, fnz = cP.fnz, NUM_PF = cP.NUM_PF;
  int pf = index/max_area;
  int bc_idx = index- pf*max_area;     

  int area_z = fnx*fny;
  if ( (bc_idx<area_z) && (pf<NUM_PF) ){
     int zj = bc_idx/fnx;
     int zi = bc_idx - zj*fnx;

     int d_out = L2G_4D(zi, zj, 0, pf, fnx, fny, fnz);
     int d_in  = L2G_4D(zi, zj, 2, pf, fnx, fny, fnz);
     int u_out = L2G_4D(zi, zj, fnz-1, pf, fnx, fny, fnz);
     int u_in  = L2G_4D(zi, zj, fnz-3, pf, fnx, fny, fnz);
     ph[d_out] = ph[d_in];
     ph[u_out] = ph[u_in];
     active_args[d_out] = active_args[d_in];
     active_args[u_out] = active_args[u_in];

  }

  int area_y = fnx*fnz;
  if ( (bc_idx<area_y) && (pf<NUM_PF) ){
     int zk = bc_idx/fnx;
     int zi = bc_idx - zk*fnx;

     int b_out = L2G_4D(zi, 0, zk, pf, fnx, fny, fnz);
     int b_in = L2G_4D(zi, fny-2, zk, pf, fnx, fny, fnz);
     int t_out = L2G_4D(zi, fny-1, zk, pf, fnx, fny, fnz);
     int t_in = L2G_4D(zi, 1, zk, pf, fnx, fny, fnz);
     ph[b_out] = ph[b_in];
     ph[t_out] = ph[t_in];
     active_args[b_out] = active_args[b_in];
     active_args[t_out] = active_args[t_in];

  }

  int area_x = fny*fnz;
  if ( (bc_idx<area_x) && (pf<NUM_PF) ){

     int zk = bc_idx/fny;
     int zj = bc_idx - zk*fny;

     int l_out = L2G_4D(0, zj, zk, pf, fnx, fny, fnz);
     int l_in = L2G_4D(fnx-2, zj, zk, pf, fnx, fny, fnz);
     int r_out = L2G_4D(fnx-1, zj, zk, pf, fnx, fny, fnz);
     int r_in = L2G_4D(1, zj, zk, pf, fnx, fny, fnz);
     ph[l_out] = ph[l_in];
     ph[r_out] = ph[r_in];
     active_args[l_out] = active_args[l_in];
     active_args[r_out] = active_args[r_in];     

  }

}

__global__ void
APTset_nofluxBC_3D(float* ph, int* active_args, int max_area){

   // dimension with R^{2D} * PF

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int fnx = cP.fnx, fny = cP.fny, fnz = cP.fnz, NUM_PF = cP.NUM_PF;
  int bcX = CP.bcX, bcY = CP.bcY, bcZ = CP.bcZ;
  int pf = index/max_area;
  int bc_idx = index- pf*max_area;     

  int area_z = fnx*fny;
  if ( (bc_idx<area_z) && (pf<NUM_PF))
  {
     int zj = bc_idx/fnx;
     int zi = bc_idx - zj*fnx;
     if (cM.processorIDZ==0)
     {
        int d_out = L2G_4D(zi, zj, 0, pf, fnx, fny, fnz);
        int d_in  = L2G_4D(zi, zj, 2, pf, fnx, fny, fnz);
        ph[d_out] = ph[d_in];
        active_args[d_out] = active_args[d_in];
     }
     if (cM.processorIDZ==cM.numProcessorZ-1)
     {
        int u_out = L2G_4D(zi, zj, fnz-1, pf, fnx, fny, fnz);
        int u_in  = L2G_4D(zi, zj, fnz-3, pf, fnx, fny, fnz);
        ph[u_out] = ph[u_in];
        active_args[u_out] = active_args[u_in];
     }
  }

  int area_y = fnx*fnz;
  if ( (bc_idx<area_y) && (pf<NUM_PF))
  {
     int zk = bc_idx/fnx;
     int zi = bc_idx - zk*fnx;
     if (cM.processorIDY==0)
     {
        int b_out = L2G_4D(zi, 0, zk, pf, fnx, fny, fnz);
        int b_in = L2G_4D(zi, 2, zk, pf, fnx, fny, fnz);
        ph[b_out] = ph[b_in];
        active_args[b_out] = active_args[b_in];
     }
     if (cM.processorIDY==cM.numProcessorY-1)
     {
        int t_out = L2G_4D(zi, fny-1, zk, pf, fnx, fny, fnz);
        int t_in = L2G_4D(zi, fny-3, zk, pf, fnx, fny, fnz);
        ph[t_out] = ph[t_in];
        active_args[t_out] = active_args[t_in];
     }
  }

  int area_x = fny*fnz;
  if ( (bc_idx<area_x) && (pf<NUM_PF))
  {

     int zk = bc_idx/fny;
     int zj = bc_idx - zk*fny;
     if (processorIDX==0)
     {
        int l_out = L2G_4D(0, zj, zk, pf, fnx, fny, fnz);
        int l_in = L2G_4D(2, zj, zk, pf, fnx, fny, fnz);
        ph[l_out] = ph[l_in];
        active_args[l_out] = active_args[l_in];
     }
     if (processorIDX==numProcessorX-1)
     {
        int r_out = L2G_4D(fnx-1, zj, zk, pf, fnx, fny, fnz);
        int r_in = L2G_4D(fnx-3, zj, zk, pf, fnx, fny, fnz);
        ph[r_out] = ph[r_in];
        active_args[r_out] = active_args[r_in]; 
     }
  }
}


// psi equation
__global__ void
APTrhs_psi(float* x, float* y, float* z, float* ph, float* ph_new, int nt, float t, int* aarg, int* aarg_new,\
       float* X, float* Y, float* Z, float* Tmac, float* u_3d, int Nx, int Ny, int Nz, int Nt, curandState* states, float* cost, float* sint){

  int C = blockIdx.x * blockDim.x + threadIdx.x; 
  int i, j, k, PF_id;
  int fnx = cP.fnx, fny = cP.fny, fnz = cP.fnz, NUM_PF = cP.NUM_PF, length = cP.length;
  G2L_3D(C, i, j, k, PF_id, fnx, fny, fnz);

  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) && (k>0) && (k<fnz-1) ) {

  //=============== load active phs ================
        int local_args[APT_NUM_PF]; // local active PF indices
        float local_phs[7][APT_NUM_PF]; // active ph for each thread ,use 7-point stencil


        int globalC, target_index, stencil, arg_index;
        for (int arg_index = 0; arg_index<NUM_PF; arg_index++){
            local_args[arg_index] = -1;
            for (stencil=0; stencil<7; stencil++){
                local_phs[stencil][arg_index] = -1.0f;
            } 
        }
        stencil = 0;
        for (int zi = -1; zi<=1; zi++){
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

       for (arg_index = 0; arg_index<NUM_PF; arg_index++){
       if (local_args[arg_index]==-1){
            aarg_new[C+arg_index*length] = -1;
            ph_new[C+arg_index*length] = -1.0f;
       }
       else{

       // start dealing with one specific PF

       PF_id = local_args[arg_index]; // global PF index to find the right orientation
       float phD=local_phs[0][arg_index], phB=local_phs[1][arg_index], phL=local_phs[2][arg_index], \
       phC=local_phs[3][arg_index], phR=local_phs[4][arg_index], phT=local_phs[5][arg_index], phU=local_phs[6][arg_index];

       float phxn = ( phR - phL ) * 0.5f;
       float phyn = ( phT - phB ) * 0.5f;
       float phzn = ( phU - phD ) * 0.5f;

       float cosa, sina, cosb, sinb;
       if (phC>LS){
       sina = sint[PF_id];
       cosa = cost[PF_id];
       sinb = sint[PF_id+cP.num_theta];
       cosb = cost[PF_id+cP.num_theta];
       }else{
       sina = 0.0f;
       cosa = 1.0f;
       sinb = 0.0f;
       cosb = 1.0f;
       }

        float A2 = kine_ani(phxn,phyn,phzn,cosa,sina,cosb,sinb);

        float diff =  phR + phL + phT + phB + phU + phD - 6*phC;
        float Tinterp = cP.G*(z[k] - cP.R*1e6 *t - 2);
        float Up = Tinterp/(cP.L_cp);  
        float repul=0.0f;
        for (int pf_id=0; pf_id<NUM_PF; pf_id++){
            
           if (pf_id!=arg_index) {
               repul += 0.25f*(local_phs[3][pf_id]+1.0f)*(local_phs[3][pf_id]+1.0f);
           }
        }
        float rhs_psi = diff * cP.hi*cP.hi + (1.0f-phC*phC)*phC \
              - cP.lamd*Up* ( (1.0f-phC*phC)*(1.0f-phC*phC) - 0.5f*OMEGA*(phC+1.0f)*repul);
        float dphi = rhs_psi / A2; 
        ph_new[C+arg_index*length] = phC  +  cP.dt * dphi; 
        if (phC  +  cP.dt * dphi <-0.9999){
            aarg_new[C+arg_index*length] = -1;
        }else{
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

__inline__ __device__ float 
nuncl_possibility(float delT, float d_delT)
{
  float slope = 0.5f*(delT-cP.undcool_mean)*(delT-cP.undcool_mean)/cP.undcool_std/cP.undcool_std;
  slope = expf(slope); 
  float density = cP.nuc_Nmax/(sqrtf(2.0f*M_PI)*cP.undcool_std) *slope*d_delT;
   // float nuc_posb = 4.0f*cP.nuc_rad*cP.nuc_rad*density; // 2D
  float nuc_posb = 8.0f*cP.nuc_rad*cP.nuc_rad*cP.nuc_rad*density; // 3D
  return nuc_posb; 
}



__global__ void
add_nucl(int* nucl_status, int cnx, int cny, int cnz, float* phi, float* alpha_m, float* x, float* y, float* z, curandState* states, 
        float dt, float t, float* X, float* Y, float* Z, float* Tmac, float* u_3d, int Nx, int Ny, int Nz, int Nt)
{
  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j, k, PF_id;
  G2L_3D(C, i, j, k, PF_id, cnx, cny, cnz);

  float Dt = Tmac[1]-Tmac[0];
  float Dx = X[1]-X[0]; 

  if ( (i<cnx) && (j<cny) && (k<cnz)) 
  {
    if (nucl_status[C]==0)
    {
      int glob_i = (2*cP.pts_cell+1)*i + cP.pts_cell;
      int glob_j = (2*cP.pts_cell+1)*j + cP.pts_cell; 
      int glob_k = (2*cP.pts_cell+1)*k + cP.pts_cell; 

      float T_cell = interp4Dtemperature(u_3d, x[glob_i] - X[0], y[glob_j] - Y[0], z[glob_k] - Z[0], t-Tmac[0], 
                                         Nx, Ny, Nz, Nt, Dx, Dt);
      float T_cell_dt = interp4Dtemperature(u_3d, x[glob_i] - X[0], y[glob_j] - Y[0], z[glob_k] - Z[0], t+dt-Tmac[0], 
                                         Nx, Ny, Nz, Nt, Dx, Dt);                                        
      float delT = cP.Tliq - T_cell_dt;
      float d_delT = T_cell - T_cell_dt;
      float nuc_posb = nuncl_possibility(delT, d_delT);

      if (curand_uniform(states+C)<nuc_posb)
      {
         printf("nucleation starts at cell no. %d \n", C);
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

    if (mpiManager->numProcessor >1)
    {
        mpiManager->createBoundaryBuffer(2*NUM_PF);
        for (auto & buffer : mpiManager->mMPIBuffer)
        {
            std::pair<float*, int> bufferPointer = buffer.second;
            cudaMalloc((void **)&(bufferPointer.first),  sizeof(float)*bufferPointer.second );
        }
    } 
    
    cudaMemcpyToSymbol(cM, mpiManager, sizeof(MPIsetting) );
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

    APTset_BC_3D<<<num_block_PF1d, blocksize_1d>>>(PFs_old, active_args_old, max_area);
    movingDomainManager->lowsl = 1;
    APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
    cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
    qois->sampleHeights(movingDomainManager->lowsl, alpha, fnx, fny, fnz);
}


void APTPhaseField::setBC(bool useLineConfig, float* ph, int* active_args)
{
    if (useLineConfig == true)
    {
        APTset_BC_3D<<<num_block_PF1d, blocksize_1d>>>(ph, active_args, max_area);
    }
    else
    {
        APTset_nofluxBC_3D<<<num_block_PF1d, blocksize_1d>>>(ph, active_args, max_area);
    }
}


void APTPhaseField::evolve()
{
    const DesignSettingData* designSetting = GetSetDesignSetting(); 
    MPIsetting* mpiManager = GetMPIManager();
    MovingDomain* movingDomainManager = new MovingDomain();

    blocksize_1d = 128;
    blocksize_2d = 128;
    num_block_2d = (length+blocksize_2d-1)/blocksize_2d;
    num_block_PF = (length*NUM_PF+blocksize_2d-1)/blocksize_2d;
    max_area = max(fnz*fny,max(fny*fnx,fnx*fnz));
    num_block_PF1d =  ( max_area*NUM_PF +blocksize_1d-1)/blocksize_1d;
    printf("block size %d, # blocks %d\n", blocksize_2d, num_block_2d); 

    if (designSetting->includeNucleation)
    {
      //int* nucl_status;
      int cnx = fnx/(2*params.pts_cell+1);
      int cny = fny/(2*params.pts_cell+1);
      int cnz = fnz/(2*params.pts_cell+1);
    }
    // create noise
    curandState* dStates;
    int period = params.noi_period;
    cudaMalloc((void **) &dStates, sizeof(curandState) * (length+period));
    
    set_minus1<<< num_block_PF, blocksize_2d>>>(PFs_old,length*NUM_PF);
    set_minus1<<< num_block_PF, blocksize_2d>>>(PFs_new,length*NUM_PF);
    APTini_PF<<< num_block_PF, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
    APTini_PF<<< num_block_PF, blocksize_2d >>>(PFs_new, phi_old, alpha_m, active_args_new);
    set_minus1<<< num_block_2d, blocksize_2d >>>(phi_old,length);

    setBC(designSetting->useLineConfig, PFs_old, active_args_old);
    setBC(designSetting->useLineConfig, PFs_new, active_args_new);

    APTrhs_psi<<< num_block_2d, blocksize_2d >>>(x_device, y_device, z_device, PFs_old, PFs_new, 0, 0, active_args_old, active_args_new,\
        Mgpu.X_mac, Mgpu.Y_mac,  Mgpu.Z_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nz, mac.Nt, dStates, Mgpu.cost, Mgpu.sint);
    cudaMemset(alpha_m, 0, sizeof(int) * length);
    APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_new, phi_old, alpha_m, active_args_new);
    cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(args_cpu, active_args_new, NUM_PF*length * sizeof(int),cudaMemcpyDeviceToHost);


    if (designSetting->useLineConfig)
    {
        movingDomainManager->allocateMovingDomain(params.num_theta, fnz);
        qois->calculateLineQoIs(params, movingDomainManager->cur_tip, alpha, 0, z, movingDomainManager->loss_area_host, movingDomainManager->move_count);
    }

    float t_cur_step;
    int kts = params.Mt/params.nts;
    printf("kts %d, nts %d\n",kts, params.nts);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    int kt = 0;

    while (kt < 1000000)
    {
        //for (int kt=0; kt<params.Mt/2; kt++){
        //for (int kt=0; kt<0; kt++){
        setBC(designSetting->useLineConfig, PFs_new, active_args_new);

        t_cur_step = (2*kt+1)*params.dt*params.tau0;
        APTrhs_psi<<< num_block_2d, blocksize_2d >>>(x_device, y_device, z_device, PFs_new, PFs_old, 2*kt+1,t_cur_step, active_args_new, active_args_old,\
        Mgpu.X_mac, Mgpu.Y_mac, Mgpu.Z_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nz, mac.Nt, dStates, Mgpu.cost, Mgpu.sint);

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

        t_cur_step = (2*kt+2)*params.dt*params.tau0;
        APTrhs_psi<<< num_block_2d, blocksize_2d >>>(x_device, y_device, z_device, PFs_old, PFs_new,  2*kt+2,t_cur_step, active_args_old, active_args_new,\
        Mgpu.X_mac, Mgpu.Y_mac, Mgpu.Z_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nz, mac.Nt, dStates, Mgpu.cost, Mgpu.sint);

        kt++;
   }


   cudaDeviceSynchronize();
   double endTime = CycleTimer::currentSeconds();
   printf("time for %d iterations: %f s\n", 2*kt, endTime-startTime);
   params.Mt = 2*kt;
   
   cudaMemset(alpha_m, 0, sizeof(int) * length);
   APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old); 
   cudaMemcpy(phi, phi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);

   if (designSetting->useLineConfig)
   {
        cudaMemcpy(alpha_i_full, d_alpha_full, fnx*fny*fnz_f * sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(alpha_i_full+movingDomainManager->move_count*fnx*fny, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
        qois->searchJunctionsOnImage(params, alpha_i_full);
   }
}









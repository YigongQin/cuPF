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
#define BLANK 0.2
#define APT_NUM_PF 5
  


__constant__ GlobalConstants cP;

__inline__ __device__ float
kine_ani(float ux, float uy, float uz, float cosa, float sina, float cosb, float sinb){

   float a_s = 1.0f + 3.0f*cP.kin_delta;
   float epsilon = -4.0f*cP.kin_delta/a_s;
   float ux2 = cosa*cosb*ux  + sina*uy + cosa*sinb*uz;
         ux2 = ux2*ux2;
   float uy2 = -sina*cosb*ux + cosa*uy - sina*sinb*uz;
         uy2 = uy2*uy2;      
   float uz2 = -sinb*ux      + 0       + cosb*uz;
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
     int d_in  = L2G_4D(zi, zj, fnz-2, pf, fnx, fny, fnz);
     int u_out = L2G_4D(zi, zj, fnz-1, pf, fnx, fny, fnz);
     int u_in  = L2G_4D(zi, zj, 1, pf, fnx, fny, fnz);
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
     int b_in = L2G_4D(zi, 2, zk, pf, fnx, fny, fnz);
     int t_out = L2G_4D(zi, fny-1, zk, pf, fnx, fny, fnz);
     int t_in = L2G_4D(zi, fny-3, zk, pf, fnx, fny, fnz);
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
     int l_in = L2G_4D(2, zj, zk, pf, fnx, fny, fnz);
     int r_out = L2G_4D(fnx-1, zj, zk, pf, fnx, fny, fnz);
     int r_in = L2G_4D(fnx-3, zj, zk, pf, fnx, fny, fnz);
     ph[l_out] = ph[l_in];
     ph[r_out] = ph[r_in];
     active_args[l_out] = active_args[l_in];
     active_args[r_out] = active_args[r_in];     

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
        float rhs_psi = diff * cP.hi*cP.hi*cP.hi + (1.0f-phC*phC)*phC \
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

void calc_qois(GlobalConstants params, QOI* q, int &cur_tip, int* alpha, int* args_cpu, float* phi, int kt, float* z, int* loss_area, int move_count){

     // cur_tip here inludes the halo
     int fnx = params.fnx, fny = params.fny, fnz = params.fnz, length = params.length, \
         num_grains = params.num_theta, NUM_PF = params.NUM_PF, all_time = params.nts+1;
     bool contin_flag = true;

     while(contin_flag){
       // at this line
        cur_tip += 1;
        int offset_z = fnx*fny*cur_tip;
        //int zeros = 0;
        for (int j=1; j<fny-1; j++){
          for (int i=1; i<fnx-1; i++){
             int C = offset_z + j*fnx + i;
             //if (alpha[C]==0){printf("find liquid at %d at line %d\n", i, cur_tip);contin_flag=false;break;}
             if (alpha[C]==0) {contin_flag=false;break;}
        }}
     }
     cur_tip -=1;
     q->tip_y[kt] = z[cur_tip];
     printf("frame %d, ntip %d, tip %f\n", kt, cur_tip+move_count, q->tip_y[kt]);
     memcpy(q->cross_sec + kt*fnx*fny, alpha + cur_tip*fnx*fny,  sizeof(int)*fnx*fny ); 

     for (int k = 1; k<fnz-1; k++){
       int offset_z = fnx*fny*k; 
       for (int j = 1; j<fny-1; j++){ 
         for (int i = 1; i<fnx-1; i++){
            int C = offset_z + fnx*j + i;
              if (alpha[C]>0){ 
                for (int time = kt; time<all_time; time++){q->tip_final[time*num_grains+alpha[C]-1] = k+move_count;} 
                q->total_area[kt*num_grains+alpha[C]-1]+=1;
                if (k > cur_tip) {q->extra_area[kt*num_grains+alpha[C]-1]+=1; }}
         }
       }
     }
     for (int j = 0; j<num_grains; j++){ 
     q->total_area[kt*num_grains+j]+=loss_area[j];}


     // find the args that have active phs greater or equal 3, copy the args to q->node_region
     int offset_z = fnx*fny*cur_tip;
     int offset_node_region = q->node_region_size/all_time*kt;
     int node_cnt = 0;
     for (int j = 1; j<fny-1; j++){ 
       for (int i = 1; i<fnx-1; i++){
          int C = offset_z + j*fnx + i;
          int neighbor_cnt = 0;
          for (int pf_id = 0; pf_id < NUM_PF; pf_id++){
             int globalC = C + pf_id*length;
             if (phi[globalC]>LS) {neighbor_cnt++;}
          } 
          if (neighbor_cnt>=3){
             q->node_region[offset_node_region + node_cnt*q->node_features] = i;
             q->node_region[offset_node_region + node_cnt*q->node_features +1] = j;
             for (int pf_id = 0; pf_id < NUM_PF; pf_id++){
                 int globalC = C + pf_id*length;
                 q->node_region[offset_node_region + node_cnt*q->node_features + 2 + pf_id] = args_cpu[globalC];
             } 
             node_cnt++;
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


void APTPhaseField::cudaSetup(params_MPI pM) {

    int num_gpus_per_node = 4;
    int device_id_innode = pM.rank % num_gpus_per_node;
    //gpu_name = cuda.select_device( )
    cudaSetDevice(device_id_innode); 
    printCudaInfo(pM.rank,device_id_innode);
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
    float* phi_cpu = new float[length * NUM_PF];

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
}





void APTPhaseField::evolve(){
  //int* nucl_status;
  
  int* left_coor = new int[params.num_theta];
  for (int i=0; i<params.num_theta; i++){left_coor[i]=1;}
 
  // allocate x, y, phi, psi, U related params
  int cnx = fnx/(2*params.pts_cell+1);
  int cny = fny/(2*params.pts_cell+1);
  int bc_len = fnx+fny+fnz;

   // create moving frame
   int move_count = 0;
   int cur_tip=1;
   int tip_front = 1;
   int tip_thres = (int) ((1-BLANK)*fnz);
   printf("max tip can go: %d\n", tip_thres); 
   float* meanx;
   cudaMalloc((void **)&meanx, sizeof(float) * fnz);
   cudaMemset(meanx,0, sizeof(float) * fnz);
  // printf(" ymax %f \n",y[fny-2] ); 
   float* meanx_host=(float*) malloc(fnz* sizeof(float));

   int* loss_area=(int*) malloc((params.num_theta)* sizeof(int));
   int* d_loss_area;
   cudaMalloc((void **)&d_loss_area, sizeof(int) * params.num_theta); 
   memset(loss_area,0,sizeof(int) * params.num_theta);
   cudaMemset(d_loss_area,0,sizeof(int) * params.num_theta); 

   // create noise
   curandState* dStates;
   int period = params.noi_period;
   cudaMalloc((void **) &dStates, sizeof(curandState) * (length+period));


   int blocksize_1d = 128;
   int blocksize_2d = 128;  // seems reduce the block size makes it a little faster, but around 128 is okay.
   int num_block_2d = (length+blocksize_2d-1)/blocksize_2d;
   int num_block_1d = (bc_len+blocksize_1d-1)/blocksize_1d;
   int num_block_PF = (length*NUM_PF+blocksize_2d-1)/blocksize_2d;
   int max_area = max(fnz*fny,max(fny*fnx,fnx*fnz));
   int num_block_PF1d =  ( max_area*NUM_PF +blocksize_1d-1)/blocksize_1d;
   printf("block size %d, # blocks %d\n", blocksize_2d, num_block_2d); 
  
   float t_cur_step;
   int kts = params.Mt/params.nts;
   printf("kts %d, nts %d\n",kts, params.nts);


   set_minus1<<< num_block_PF, blocksize_2d>>>(PFs_old,length*NUM_PF);
   set_minus1<<< num_block_PF, blocksize_2d>>>(PFs_new,length*NUM_PF);
   APTini_PF<<< num_block_PF, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
   APTini_PF<<< num_block_PF, blocksize_2d >>>(PFs_new, phi_old, alpha_m, active_args_new);
   set_minus1<<< num_block_2d, blocksize_2d >>>(phi_old,length);

     //cudaDeviceSynchronize();
   APTset_BC_3D<<<num_block_PF1d, blocksize_1d>>>(PFs_new, active_args_new, max_area);
   APTset_BC_3D<<<num_block_PF1d, blocksize_1d>>>(PFs_old, active_args_old, max_area);

   APTrhs_psi<<< num_block_2d, blocksize_2d >>>(x_device, y_device, z_device, PFs_old, PFs_new, 0, 0, active_args_old, active_args_new,\
    Mgpu.X_mac, Mgpu.Y_mac,  Mgpu.Z_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nz, mac.Nt, dStates, Mgpu.cost, Mgpu.sint);
   cudaMemset(alpha_m, 0, sizeof(int) * length);
   APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_new, phi_old, alpha_m, active_args_new);
   cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
   cudaMemcpy(args_cpu, active_args_new, NUM_PF*length * sizeof(int),cudaMemcpyDeviceToHost);
   cudaMemcpy(phi_cpu, PFs_new, NUM_PF*length * sizeof(float),cudaMemcpyDeviceToHost);
   calc_qois(params, q, cur_tip, alpha, args_cpu, phi_cpu, 0, z, loss_area, move_count);
   cudaDeviceSynchronize();
   double startTime = CycleTimer::currentSeconds();
   for (int kt=0; kt<params.Mt/2; kt++){
   //for (int kt=0; kt<0; kt++){
     APTset_BC_3D<<<num_block_PF1d, blocksize_1d>>>(PFs_new, active_args_new, max_area);

     t_cur_step = (2*kt+1)*params.dt*params.tau0;
     APTrhs_psi<<< num_block_2d, blocksize_2d >>>(x_device, y_device, z_device, PFs_new, PFs_old, 2*kt+1,t_cur_step, active_args_new, active_args_old,\
        Mgpu.X_mac, Mgpu.Y_mac, Mgpu.Z_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nz, mac.Nt, dStates, Mgpu.cost, Mgpu.sint);
 
     APTset_BC_3D<<<num_block_PF1d, blocksize_1d>>>(PFs_old, active_args_old, max_area);

    if ( (2*kt+2)%kts==0) {
             //tip_mvf(&cur_tip,phi_new, meanx, meanx_host, fnx,fny);
             cudaMemset(alpha_m, 0, sizeof(int) * length);
             APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
             cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost); 
             cudaMemcpy(args_cpu, active_args_old, NUM_PF*length * sizeof(int),cudaMemcpyDeviceToHost);
             cudaMemcpy(phi_cpu, PFs_old, NUM_PF*length * sizeof(float),cudaMemcpyDeviceToHost);
             cudaMemcpy(loss_area, d_loss_area, params.num_theta * sizeof(int),cudaMemcpyDeviceToHost);
             cudaMemcpy(z, z_device, fnz * sizeof(int),cudaMemcpyDeviceToHost); 
             //QoIs based on alpha field
             cur_tip=0;
             calc_qois(params, q, cur_tip, alpha, args_cpu, phi_cpu, (2*kt+2)/kts, z, loss_area, move_count);
          }

   if ( (2*kt+2)%TIPP==0) {
             tip_mvf(&tip_front, PFs_old, meanx, meanx_host, fnx,fny,fnz,NUM_PF);
             while (tip_front >=tip_thres){
                APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old);
                APTmove_frame<<< num_block_PF, blocksize_2d >>>(PFs_new, active_args_new, z_device2, PFs_old, active_args_old, z_device, alpha_m, d_alpha_full, d_loss_area, move_count);
                APTcopy_frame<<< num_block_PF, blocksize_2d >>>(PFs_new, active_args_new, z_device2, PFs_old, active_args_old, z_device);
                move_count +=1;
                tip_front-=1;

             }

          APTset_BC_3D<<<num_block_PF1d, blocksize_1d>>>(PFs_old, active_args_old, max_area);
          }

     t_cur_step = (2*kt+2)*params.dt*params.tau0;
     APTrhs_psi<<< num_block_2d, blocksize_2d >>>(x_device, y_device, z_device, PFs_old, PFs_new,  2*kt+2,t_cur_step, active_args_old, active_args_new,\
        Mgpu.X_mac, Mgpu.Y_mac, Mgpu.Z_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nz, mac.Nt, dStates, Mgpu.cost, Mgpu.sint);


   }


   cudaDeviceSynchronize();
   double endTime = CycleTimer::currentSeconds();
   printf("time for %d iterations: %f s\n", params.Mt, endTime-startTime);

   
   cudaMemset(alpha_m, 0, sizeof(int) * length);
   APTcollect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, active_args_old); 
   cudaMemcpy(phi, phi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
   cudaMemcpy(alpha_i_full, d_alpha_full, fnx*fny*fnz_f * sizeof(int),cudaMemcpyDeviceToHost);
   cudaMemcpy(alpha_i_full+move_count*fnx*fny, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
 //  calc_frac(alpha, fnx, fny, fnz, params.nts, params.num_theta, tip_y, frac, z, aseq, ntip, left_coor);

}









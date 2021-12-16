#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <mpi.h>
#include "CycleTimer.h"
#include "include_struct.h"
using namespace std;
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCKSIZE BLOCK_DIM_X*BLOCK_DIM_Y
#define HALO 1//halo in global region
#define REAL_DIM_X 14 //BLOCK_DIM_X-2*HALO
#define REAL_DIM_Y 14 //BLOCK_DIM_Y-2*HALO
#define LS -0.995
#define ACR 1e-5
#define NBW 1
#define NUM_PF 8
#define OMEGA 200
#define ZERO 0

#define TIPP 20
#define BLANK 0.3

void printCudaInfo(int rank, int i);
extern float toBW(int bytes, float sec);

  
struct BC_buffs{
   // R>L, T>B
   // the dimension of each is ha_wd*num_fields*length
   float* sendR; 
   float* sendL;
   float* sendT;
   float* sendB;
   float* sendRT;
   float* sendRB; 
   float* sendLT;
   float* sendLB;
   float* recvR;
   float* recvL;
   float* recvT;
   float* recvB;
   float* recvRT;
   float* recvRB;
   float* recvLT;
   float* recvLB;
   


};



__constant__ GlobalConstants cP;


void print2d(float* array, int fnx, int fny){

   int length = fnx*fny;
   float* cpu_array = new float[fnx*fny];

   cudaMemcpy(cpu_array, array, length * sizeof(float),cudaMemcpyDeviceToHost);

   for (int i=0; i<length; i++){
       if (i%fnx==0) printf("\n");
       printf("%4.2f ",cpu_array[i]);
   }

}


// Device codes 

// boundary condition
// only use this function to access the boundary points, 
// other functions return at the boundary


__global__ void
set_BC_mpi_y(float* ph, int fnx, int fny, \
        int px, int py, int nprocx, int nprocy, int ha_wd){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int PF_id = index/fnx;    
  int pf_index = index- PF_id*fnx; 
  int offset = PF_id*fnx*fny;
  if ( (pf_index<fnx) && (PF_id<NUM_PF) ){
  // orignially it is 0,2; fny-1, fny-3
  // now ha_wd-1, ha_wd+1; fny-ha_wd, fny-ha_wd-2
      if (py==0) {
          int b_out = pf_index+(ha_wd-1)*fnx + offset;
          int b_in = pf_index+(ha_wd+1)*fnx + offset;

            //ps[b_out] = ps[b_in];
            ph[b_out] = ph[b_in];
            //U[b_out] = U[b_in];
            //dpsi[b_out] = dpsi[b_in];
            //ph2[b_out] = ph2[b_in];
            }
       if (py==nprocy-1) {     
          int t_out = pf_index+(fny-ha_wd)*fnx + offset;
          int t_in = pf_index+(fny-ha_wd-2)*fnx + offset;        
           // ps[t_out] = ps[t_in];
            ph[t_out] = ph[t_in];
           // U[t_out] = U[t_in];
           // dpsi[t_out] = dpsi[t_in];
           // ph2[t_out] = ph2[t_in];
           }
  }

}
__global__ void
set_BC_mpi_x(float* ph, int fnx, int fny, \
        int px, int py, int nprocx, int nprocy, int ha_wd){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int PF_id = index/fny;
  int pf_index = index- PF_id*fny; 
  int offset = PF_id*fnx*fny;
  if ( (pf_index<fny) && (PF_id<NUM_PF) ){
  // orignially it is 0,2; fnx-1, fnx-3
  // now ha_wd-1, ha_wd+1; fnx-ha_wd, fnx-ha_wd-2  
      if (px==0) {
      
         int l_out = pf_index*fnx + ha_wd-1 + offset;
         int l_in = pf_index*fnx + ha_wd+1 + offset;
          //  ps[l_out] = ps[l_in];
            ph[l_out] = ph[l_in];
          //  U[l_out] = U[l_in];
          //  dpsi[l_out] = dpsi[l_in];
          //  ph2[l_out] = ph2[l_in];
      }
      if (px==nprocx-1){
      
         int r_out = pf_index*fnx + fnx-ha_wd + offset;
         int r_in = pf_index*fnx + fnx-ha_wd-2 + offset;    
          //  ps[r_out] = ps[r_in];
            ph[r_out] = ph[r_in];
          //  U[r_out] = U[r_in];
          // dpsi[r_out] = dpsi[r_in];
          //  ph2[r_out] = ph2[r_in];
      }  
  }
}


__global__ void
collect2(float *ptr[], BC_buffs BC, int fnx, int fny, int num_fields, int max_len){

  // parallism we have: ha_wd*max(nx,ny)
  //  int Lx = num_fields*hd*nx;
  //  int Ly = num_fields*hd*ny;
  //  int Lxy = num_fields*hd*hd;
  //float *ptr[5]={ps,ph,U,dpsi,ph2};
  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int hd = cP.ha_wd;

  int fid = C/(max_len*hd);
  int index = C-fid*max_len*hd;
  int i = index/hd;  // range [0,max]
  int j = index-i*hd;  // range [0,hd]
  int nx = fnx-2*hd; // actual size.
  int ny = fny-2*hd;
  int stridexy = hd*hd;
  if ( (i<ny) && (j<hd) && (fid<num_fields)){
      // left,right length ny
      int field_indexL = j+hd+(i+hd)*fnx;
      int field_indexR = j+nx+(i+hd)*fnx;
      int stridey = hd*ny; // stride is 
     // for (int fid=0; fid<5; fid++) {
        BC.sendL[index+fid*stridey] = ptr[fid][field_indexL];
        BC.sendR[index+fid*stridey] = ptr[fid][field_indexR];
       
     // }

  }
  
  if ( (i<nx) && (j<hd) && (fid<num_fields)){
       // up,bottom, length nx
      int field_indexB = i+hd+(j+hd)*fnx;
      int field_indexT = i+hd+(j+ny)*fnx;
      int stridex = hd*nx; // stride is 
      //for (int fid=0; fid<5; fid++) {
       BC.sendB[index+fid*stridex] = ptr[fid][field_indexB];
       BC.sendT[index+fid*stridex] = ptr[fid][field_indexT];
         if (i<hd){
         int field_indexLB = i+hd+(j+hd)*fnx;
         int field_indexLT = i+hd+(j+ny)*fnx;
         int field_indexRB = i+nx+(j+hd)*fnx;
         int field_indexRT = i+nx+(j+ny)*fnx;
         BC.sendLB[index+fid*stridexy] = ptr[fid][field_indexLB];
         BC.sendLT[index+fid*stridexy] = ptr[fid][field_indexLT];
         BC.sendRB[index+fid*stridexy] = ptr[fid][field_indexRB];
         BC.sendRT[index+fid*stridexy] = ptr[fid][field_indexRT];
       
       }
      //}
  
  }

  
}



__global__ void
distribute2(float *ptr[], BC_buffs BC, int fnx, int fny, int num_fields, int max_len){

  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int hd = cP.ha_wd;

  int fid = C/(max_len*hd);
  int index = C-fid*max_len*hd;
  //float *ptr[5]={ps,ph,U,dpsi,ph2};
  int i = index/hd;  // range [0,max]
  int j = index-i*hd;  // range [0,hd]
  int nx = fnx-2*hd; // actual size.
  int ny = fny-2*hd;
  int stridexy = hd*hd;

  if ( (i<ny) && (j<hd) && (fid<num_fields)){
      // left,right length ny
      int field_indexL = j+(i+hd)*fnx;
      int field_indexR = j+nx+hd+(i+hd)*fnx;
      int stridey = hd*ny; // stride is
     // for (int fid=0; fid<5; fid++) {
      ptr[fid][field_indexL] = BC.recvL[index+fid*stridey];
      ptr[fid][field_indexR] = BC.recvR[index+fid*stridey];

      //}

  }
  if ( (i<nx) && (j<hd) && (fid<num_fields)){
       // up,bottom, length nx
      int field_indexB = i+hd+(j)*fnx;
      int field_indexT = i+hd+(j+ny+hd)*fnx;
      int stridex = hd*nx; // stride is
     // for (int fid=0; fid<5; fid++) {
       ptr[fid][field_indexB] = BC.recvB[index+fid*stridex];
       ptr[fid][field_indexT] = BC.recvT[index+fid*stridex];
         if (i<hd){
         int field_indexLB = i+(j)*fnx;
         int field_indexLT = i+(j+ny+hd)*fnx;
         int field_indexRB = i+nx+hd+(j)*fnx;
         int field_indexRT = i+nx+hd+(j+ny+hd)*fnx;
         ptr[fid][field_indexLB] = BC.recvLB[index+fid*stridexy];
         ptr[fid][field_indexLT] = BC.recvLT[index+fid*stridexy];
         ptr[fid][field_indexRB] = BC.recvRB[index+fid*stridexy];
         ptr[fid][field_indexRT] = BC.recvRT[index+fid*stridexy];

       }
      //}

  }
}



__global__ void
XY_2D_interp(float* x, float* y, float* X, float* Y, float* u_2d, float* u_m, int Nx, int Ny, int fnx, int fny){

   int C = blockIdx.x * blockDim.x + threadIdx.x;
   int j=C/fnx;
   int i=C-j*fnx;

   float Dx = X[1]-X[0]; // (X[Nx-1]-X[0]) / (Nx-1)
   float Dy = Y[1]-Y[0];

   if ( (i<fnx) && (j<fny) ){
      //printf("%d ",i);
      int kx = (int) (( x[i] - X[0] )/Dx);
      float delta_x = ( x[i] - X[0] )/Dx - kx;
         //printf("%f ",delta_x);
      int ky = (int) (( y[j] - Y[0] )/Dy);
      float delta_y = ( y[j] - Y[0] )/Dy - ky;
      int offset = kx + ky*Nx;
      u_m[C] =  (1.0f-delta_x)*(1.0f-delta_y)*u_2d[ offset ] + (1.0f-delta_x)*delta_y*u_2d[ offset+Nx ] \
               +delta_x*(1.0f-delta_y)*u_2d[ offset+1 ] +   delta_x*delta_y*u_2d[ offset+Nx+1 ];
      }

}




__global__ void
init_rand_num(curandState *state, int seed_val, int len_plus){

  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id<len_plus) curand_init(seed_val, id, 0, &state[id]);

}

__global__ void
gen_rand_num(curandState *state, float* random_nums, int len_plus){

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id<len_plus) random_nums[id] = curand_uniform(state+id)-0.5;

}

// anisotropy functions

__inline__ __device__ float
kine_ani(float ux, float uz, float cosa, float sina){

   float a_s = 1.0f + 3.0f*cP.kin_delta;
   float epsilon = -4.0f*cP.kin_delta/a_s;
   float ux2 = cosa*ux + sina*uz;
         ux2 = ux2*ux2;
   float uz2 = -sina*ux + cosa*uz;
         uz2 = uz2*uz2;
   float MAG_sq = (ux2 + uz2);
   float MAG_sq2= MAG_sq*MAG_sq;
   if (MAG_sq > cP.eps){
         return a_s*( 1.0f + epsilon*(ux2*ux2 + uz2*uz2) / MAG_sq2);}
   else {return 1.0f;}
}

__inline__ __device__ float
atheta(float ux, float uz, float cosa, float sina){
  
   float ux2 = cosa*ux + sina*uz;
         ux2 = ux2*ux2;
   float uz2 = -sina*ux + cosa*uz;
         uz2 = uz2*uz2;
   float MAG_sq = (ux2 + uz2);
   float MAG_sq2= MAG_sq*MAG_sq;
   if (MAG_sq > cP.eps){
         return cP.a_s*( 1.0f + cP.epsilon*(ux2*ux2 + uz2*uz2) / MAG_sq2);}
   else {return 1.0f;}
}


__inline__ __device__ float
aptheta(float ux, float uz, float cosa, float sina){

   float uxr = cosa*ux + sina*uz;
   float ux2 = uxr*uxr;
   float uzr = -sina*ux + cosa*uz;
   float uz2 = uzr*uzr;
   float MAG_sq = (ux2 + uz2);
   float MAG_sq2= MAG_sq*MAG_sq;
   if (MAG_sq > cP.eps){
         return -cP.a_12*uxr*uzr*(ux2 - uz2) / MAG_sq2;}
   else {return 0.0f;}
}

__device__ float 
nuncl_possibility(float delT, float d_delT  ){

  float slope = -0.5f*(delT-cP.undcool_mean)*(delT-cP.undcool_mean)/cP.undcool_std/cP.undcool_std;
//  float slope = -0.5f*(delT-cP.undcool_mean)/cP.undcool_std;
  slope = expf(slope); 
  float density = cP.nuc_Nmax/(sqrtf(2.0f*M_PI)*cP.undcool_std) *slope*d_delT;
  float nuc_posb = 4.0f*cP.nuc_rad*cP.nuc_rad*density;
  //int pts_cell;


  return nuc_posb; 

}

__global__ void
init_nucl_status(float* ph, int* nucl_status, int cnx, int cny, int fnx){
  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int j=C/cnx;
  int i=C-j*cnx;

  if ( (i<cnx) && (j<cny) ) {
   
      int glob_i = (2*cP.pts_cell+1)*i + cP.pts_cell;
      int glob_j = (2*cP.pts_cell+1)*j + cP.pts_cell;
      int glob_C = glob_j*fnx + glob_i;   
      if (ph[glob_C]>LS){
          nucl_status[C] = 1;
      } 
      else {nucl_status[C] = 0;}

  }
}

__global__ void
add_nucl(int* nucl_status, int cnx, int cny, float* ph, int* alpha_m, float* x, float* y, int fnx, int fny, curandState* states, \
        float dt, float t, float* X, float* Y, float* Tmac, float* u_3d, int Nx, int Ny, int Nt){

  // fnx = (2*cP.) 
  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int j=C/cnx;
  int i=C-j*cnx;
   float Dt = Tmac[1]-Tmac[0];
   float Dx = X[1]-X[0]; // (X[Nx-1]-X[0]) / (Nx-1)
   float Dy = Y[1]-Y[0];
  if ( (i<cnx) && (j<cny) ) {
    if (nucl_status[C]==0){
      int glob_i = (2*cP.pts_cell+1)*i + cP.pts_cell;
      int glob_j = (2*cP.pts_cell+1)*j + cP.pts_cell; 
      int glob_C = glob_j*fnx + glob_i;

     for (int pf_id=0; pf_id<NUM_PF; pf_id++) { if (ph[glob_C+pf_id*fnx*fny]>LS) {nucl_status[C]=1;} }
     if (nucl_status[C]==0) {  
      int kx = (int) (( x[glob_i] - X[0] )/Dx);
      float delta_x = ( x[glob_i] - X[0] )/Dx - kx;
         //printf("%f ",delta_x);
      int ky = (int) (( y[glob_j] - Y[0] )/Dy);
      float delta_y = ( y[glob_j] - Y[0] )/Dy - ky;
      //printf("%d ",kx);
      if (kx==Nx-1) {kx = Nx-2; delta_x =1.0f;}
      if (ky==Ny-1) {ky = Ny-2; delta_y =1.0f;}

      int kt = (int) ((t-Tmac[0])/Dt);
      float delta_t = (t-Tmac[0])/Dt-kt;
      if (kt==Nt-1) {kt = Nt-2; delta_t =1.0f;}
      int offset =  kx + ky*Nx + kt*Nx*Ny;
      int offset_n =  kx + ky*Nx + (kt+1)*Nx*Ny;
      float T_cell = ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset ] + (1.0f-delta_x)*delta_y*u_3d[ offset+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset+1 ] +   delta_x*delta_y*u_3d[ offset+Nx+1 ] )*(1.0f-delta_t) + \
             ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset_n ] + (1.0f-delta_x)*delta_y*u_3d[ offset_n+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset_n+1 ] +   delta_x*delta_y*u_3d[ offset_n+Nx+1 ] )*delta_t;

      int kt1 = (int) ((t+dt-Tmac[0])/Dt);
      float delta_t1 = (t+dt-Tmac[0])/Dt-kt1;
      if (kt1==Nt-1) {kt1 = Nt-2; delta_t1 =1.0f;}
      int offset1 =  kx + ky*Nx + kt1*Nx*Ny;
      int offset1_n =  kx + ky*Nx + (kt1+1)*Nx*Ny;
      float T_cell1 = ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset1 ] + (1.0f-delta_x)*delta_y*u_3d[ offset1+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset1+1 ] +   delta_x*delta_y*u_3d[ offset1+Nx+1 ] )*(1.0f-delta_t1) + \
             ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset1_n ] + (1.0f-delta_x)*delta_y*u_3d[ offset1_n+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset1_n+1 ] +   delta_x*delta_y*u_3d[ offset1_n+Nx+1 ] )*delta_t1;
      float delT = cP.Tliq - T_cell;
      float d_delT = T_cell - T_cell1;
      if (delT>cP.undcool_mean){
        float nuc_posb = nuncl_possibility(delT, d_delT);
        if (curand_uniform(states+C)< nuc_posb){
         printf("possibility, on curve %f\n", nuc_posb);
       // if (curand_uniform(states+C)< cP.nuc_Nmax ){ //nuc_posb){
        // printf("possibility, constant %f\n", cP.nuc_Nmax);
           // start to render circles
         //float rand_angle = curand_uniform(states+C)*(-M_PI/2.0f);
         int rand_PF = curand(states+C)%NUM_PF;
         int offset_rand = rand_PF*fnx*fny;
         printf("time %f, nucleation starts at cell no. %d get PF no. %d\n", t, C, rand_PF);
         for (int locj=-cP.pts_cell;locj<=cP.pts_cell;locj++){
              for (int loci=-cP.pts_cell;loci<=cP.pts_cell;loci++){
                     int os_loc = locj*fnx+loci+glob_C+offset_rand;
                     float dist_C = cP.dx*( (1.0f+cP.pts_cell)/2.0f - sqrtf(loci*loci + locj*locj) );
                     ph[os_loc] = tanhf( dist_C /cP.sqrt2 );
                  //   alpha_m[os_loc] = rand_angle;
              }
         }
         nucl_status[C] = 1;
        } 
     }
     }
    }
  }
}

// psi equation
__global__ void
rhs_psi(float* ph, float* ph_new, float* x, float* y, int fnx, int fny, int nt, \
       float t, float* X, float* Y, float* Tmac, float* u_3d, int Nx, int Ny, int Nt, curandState* states, float* cost, float* sint ){

  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int PF_id = C/(fnx*fny);
  int pf_C = C - PF_id*fnx*fny;  // local C in every PF
  int j=pf_C/fnx; 
  int i=pf_C-j*fnx;
  // macros
   float Dt = Tmac[1]-Tmac[0];
   int kt = (int) ((t-Tmac[0])/Dt);
  // printf("%d ",kt);
   float delta_t = (t-Tmac[0])/Dt-kt;
   //printf("%f ",Dt);
   float Dx = X[1]-X[0]; // (X[Nx-1]-X[0]) / (Nx-1)
   float Dy = Y[1]-Y[0];
  // if the points are at boundary, return
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) &&(PF_id<NUM_PF) ) {
       // find the indices of the 8 neighbors for center
      //if ( (ph[C]<1.0f) && (ph[C]>-1.0f) ){
       //if (C==1000){printf("find");}
       int R=C+1;
       int L=C-1;
       int T=C+fnx;
       int B=C-fnx;
       //float alpha = theta_arr[PF_id+1];
       float cosa, sina;
       if (ph[C]>LS){
       sina = sint[PF_id+1];
       cosa = cost[PF_id+1];
       }else{
       sina = 0.0f;
       cosa = 1.0f;
       }
       // first checkout the anisotropy 
        float phxn = ( ph[R] - ph[L] ) * 0.5f;
        float phzn = ( ph[T] - ph[B] ) * 0.5f;

       float A2;
       float ux2 = cosa*phxn + sina*phzn;
         ux2 = ux2*ux2;
       float uz2 = -sina*phxn + cosa*phzn;
         uz2 = uz2*uz2;
       float MAG_sq = (ux2 + uz2);
       float MAG_sq2= MAG_sq*MAG_sq;
       if (MAG_sq > cP.eps){
          A2 =  cP.a_s*( 1.0f + cP.epsilon*(ux2*ux2 + uz2*uz2) / MAG_sq2);}
       else {A2 = 1.0f;}        //float A2 = atheta(phxn,phzn,cosa,sina);
        //float Ak2 = kine_ani(phxn,phzn,cosa,sina);

        float Ak2 = kine_ani(phxn,phzn,cosa,sina);
        A2 = A2*Ak2;

        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================


        float phipjp=( ph[C] + ph[R] + ph[T] + ph[T+1] ) * 0.25f;
        float phipjm=( ph[C] + ph[R] + ph[B] + ph[B+1] ) * 0.25f;
        float phimjp=( ph[C] + ph[L] + ph[T-1] + ph[T] ) * 0.25f;
        float phimjm=( ph[C] + ph[L] + ph[B-1] + ph[B] ) * 0.25f;
        
        // ============================
        // right edge flux
        // ============================
        float phx = ph[R]-ph[C];
        float phz = phipjp - phipjm;

        float A  = atheta( phx,phz,cosa,sina);
        float Ap = aptheta(phx,phz,cosa,sina);
        float JR = A * ( A*phx - Ap*phz );
        
        // ============================
        // left edge flux
        // ============================
        phx = ph[C]-ph[L];
        phz = phimjp - phimjm; 

        A  = atheta( phx,phz,cosa,sina);
        Ap = aptheta(phx,phz,cosa,sina);
        float JL = A * ( A*phx - Ap*phz );
        
        // ============================
        // top edge flux
        // ============================
        phx = phipjp - phimjp;
        phz = ph[T]-ph[C];

        A  = atheta( phx,phz,cosa,sina);
        Ap = aptheta(phx,phz,cosa,sina);
        float JT = A * ( A*phz + Ap*phx );

        // ============================
        // bottom edge flux
        // ============================
        phx = phipjm - phimjm;
        phz = ph[C]-ph[B];

        A  = atheta( phx,phz,cosa,sina);
        Ap = aptheta(phx,phz,cosa,sina);
        float JB = A * ( A*phz + Ap*phx );


        /*# =============================================================
        #
        # 3. double well (transformed): sqrt2 * phi + nonlinear terms
        #
        # =============================================================*/
        //float Up = (y[j]/cP.W0 - cP.R_tilde * (nt*cP.dt) )/cP.lT_tilde;
      int kx = (int) (( x[i] - X[0] )/Dx);
      float delta_x = ( x[i] - X[0] )/Dx - kx;
         //printf("%f ",delta_x);
      int ky = (int) (( y[j] - Y[0] )/Dy);
      float delta_y = ( y[j] - Y[0] )/Dy - ky;
      //printf("%d ",kx);
      if (kx==Nx-1) {kx = Nx-2; delta_x =1.0f;}
      if (ky==Ny-1) {ky = Ny-2; delta_y =1.0f;}
      if (kt==Nt-1) {kt = Nt-2; delta_t =1.0f;}
      int offset =  kx + ky*Nx + kt*Nx*Ny;
      int offset_n =  kx + ky*Nx + (kt+1)*Nx*Ny;
      //if (offset_n>Nx*Ny*Nt-1-1-Nx) printf("%d, %d, %d, %d  ", i,j,kx,ky);
     // printf("%d ", Nx);
      float Tinterp= ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset ] + (1.0f-delta_x)*delta_y*u_3d[ offset+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset+1 ] +   delta_x*delta_y*u_3d[ offset+Nx+1 ] )*(1.0f-delta_t) + \
             ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset_n ] + (1.0f-delta_x)*delta_y*u_3d[ offset_n+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset_n+1 ] +   delta_x*delta_y*u_3d[ offset_n+Nx+1 ] )*delta_t;

        float Up = (Tinterp-cP.Tmelt)/(cP.L_cp);  //(y[j]/cP.W0 - cP.R_tilde * (nt*cP.dt) )/cP.lT_tilde;
       // float Up = (Tinterp-cP.Ti)/(cP.c_infm/cP.k)/(1.0-cP.k);  //(y[j]/cP.W0 - cP.R_tilde * (nt*cP.dt) )/cP.lT_tilde;
        float repul=0.0f;
        for (int pf_id=0; pf_id<NUM_PF; pf_id++){
            
           if (pf_id!=PF_id) {
               int overlap_id = pf_C + pf_id*fnx*fny;
               repul += 0.25f*(ph[overlap_id]+1.0f)*(ph[overlap_id]+1.0f);
           }
        }
        float rhs_psi = ((JR-JL) + (JT-JB) ) * cP.hi*cP.hi + \
                  (1.0f-ph[C]*ph[C])*( ph[C] - cP.lamd*(1.0f-ph[C]*ph[C])*( Up) )  - 0.5f*OMEGA*(ph[C]+1.0f)*repul;

      //# =============================================================
        //#
        //# 4. dpsi/dt term
       // #
        //# =============================================================
        //float tp = (1.0f-(1.0f-cP.k)*Up);
        //float tau_psi;
        //if (tp >= cP.k){tau_psi = tp*A2;}
              // else {tau_psi = cP.k*A2;}
        
        //dpsi[C] = rhs_psi / tau_psi; 
        float dphi = rhs_psi / A2; //tau_psi;
        //float rand;
        //if ( ( ph[C]>-0.995 ) && ( ph[C]<0.995 ) ) {rand= cP.dt_sqrt*cP.hi*cP.eta*(curand_uniform(states+C)-0.5);}
        //else {rand = 0.0f;}      //  ps_new[C] = ps[C] +  cP.dt * dpsi[C];
        //int new_noi_loc = nt%cP.noi_period;*cP.seed_val)%(fnx*fny);
        ph_new[C] = ph[C]  +  cP.dt * dphi; // + rand; //cP.dt_sqrt*cP.hi*cP.eta*rnd[C+new_noi_loc];
        //if ( (ph_new[C]<-1.0f)||(ph_new[C]>1.0f) ) printf("blow up\n");
        //if (C==1000){printf("%f ",ph_new[C]);}

     }
} 

// U equation

__global__ void
set_minus1(float* u, int size){

     int index = blockIdx.x * blockDim.x + threadIdx.x;
     if(index<size) u[index] = -1.0f;

}


__global__ void
collect_PF(float* PFs, float* phi, int* alpha_m, int length, int* argmax){

   int C = blockIdx.x * blockDim.x + threadIdx.x;
 //  int PF_id = index/(length);
 //  int C = index - PF_id*length;
   
  //if ( (C<length) && (PF_id<NUM_PF) ) {

    //if (C==513*717+716){printf("PF_id %d, values %f\n", PF_id, PFs[index]);}
   if (C<length){
   // for loop to find the argmax of the phase field
     for (int PF_id=0; PF_id<NUM_PF; PF_id++){
       int loc = C + length*PF_id; 
       int max_loc = C + length*argmax[C];
       if (PFs[loc]>PFs[max_loc]) {argmax[C]=PF_id;}
     }
    
   int max_loc_f = C + length*argmax[C]; 
   if (PFs[max_loc_f]>LS){
      phi[C] = PFs[max_loc_f]; 
      alpha_m[C] = argmax[C] +1;
    }
   }



}


__global__ void
ini_PF(float* PFs, float* phi, int* alpha_m, int length){

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int PF_id = index/(length);
   int C = index - PF_id*length;

 // if (PF_id==1){
  if ( (C<length) && (PF_id<NUM_PF) ) {
    if ( (phi[C]>LS) && (PF_id== (alpha_m[C]-1)) ){
  //  if ( (phi[C]>LS) && (PF_id== 0)){ //(alpha_m[C]-1)) ){
      PFs[index] = phi[C];
    }
 // }
  }
}



void allocate_mpi_buffs(GlobalConstants params, BC_buffs *DNS, int fnx, int fny){

    int num_fields = 5;
    int hd = params.ha_wd;
    int nx = fnx-2*hd;
    int ny = fny-2*hd;     
    int Lx = num_fields*hd*nx;
    int Ly = num_fields*hd*ny;
    int Lxy = num_fields*hd*hd;
 
    cudaMalloc((void **)&(DNS->sendR),  sizeof(float) * Ly);
    cudaMalloc((void **)&(DNS->sendL),  sizeof(float) * Ly);
    cudaMalloc((void **)&(DNS->sendT),  sizeof(float) * Lx);
    cudaMalloc((void **)&(DNS->sendB),  sizeof(float) * Lx);    
    cudaMalloc((void **)&(DNS->recvR),  sizeof(float) * Ly);
    cudaMalloc((void **)&(DNS->recvL),  sizeof(float) * Ly);
    cudaMalloc((void **)&(DNS->recvT),  sizeof(float) * Lx);
    cudaMalloc((void **)&(DNS->recvB),  sizeof(float) * Lx);  

    cudaMalloc((void **)&(DNS->sendRT),  sizeof(float) * Lxy);
    cudaMalloc((void **)&(DNS->sendLT),  sizeof(float) * Lxy);
    cudaMalloc((void **)&(DNS->sendRB),  sizeof(float) * Lxy);
    cudaMalloc((void **)&(DNS->sendLB),  sizeof(float) * Lxy);    
    cudaMalloc((void **)&(DNS->recvRT),  sizeof(float) * Lxy);
    cudaMalloc((void **)&(DNS->recvLT),  sizeof(float) * Lxy);
    cudaMalloc((void **)&(DNS->recvRB),  sizeof(float) * Lxy);
    cudaMalloc((void **)&(DNS->recvLB),  sizeof(float) * Lxy); 



}



void exchange_BC(MPI_Comm comm, BC_buffs BC, int hd, int fnx, int fny, int nt, int rank, int px, int py, int nprocx, int nprocy){

    int ntag = 8*nt;
    int num_fields = 5;
    int nx = fnx-2*hd;
    int ny = fny-2*hd;
    int Lx = num_fields*hd*nx;
    int Ly = num_fields*hd*ny;
    int Lxy = num_fields*hd*hd;


    if ( px < nprocx-1 ) //right send
          {MPI_Send(BC.sendR, Ly, MPI_FLOAT, rank+1, ntag+1, comm);}
    if ( px > 0 )
          {MPI_Recv(BC.recvL, Ly, MPI_FLOAT, rank-1, ntag+1, comm, MPI_STATUS_IGNORE);}
    if ( px > 0 ) //left send
          {MPI_Send(BC.sendL, Ly, MPI_FLOAT, rank-1, ntag+2, comm);}
    if ( px < nprocx-1 )
          {MPI_Recv(BC.recvR, Ly, MPI_FLOAT, rank+1, ntag+2, comm, MPI_STATUS_IGNORE);}


    if ( py < nprocy-1 ) //top send
          {MPI_Send(BC.sendT, Lx, MPI_FLOAT, rank+nprocx, ntag+3, comm);}
    if ( py>0 )
          {MPI_Recv(BC.recvB, Lx, MPI_FLOAT, rank-nprocx, ntag+3, comm, MPI_STATUS_IGNORE);}
    if ( py >0 ) //bottom send
          {MPI_Send(BC.sendB, Lx, MPI_FLOAT, rank-nprocx, ntag+4, comm);}
    if ( py < nprocy -1 )
          {MPI_Recv(BC.recvT, Lx, MPI_FLOAT, rank+nprocx, ntag+4, comm, MPI_STATUS_IGNORE);}


    if ( px < nprocx-1 and py < nprocy-1)
          {MPI_Send(BC.sendRT, Lxy, MPI_FLOAT, rank+1+nprocx, ntag+5, comm);}
    if ( px > 0 and py > 0 )
          {MPI_Recv(BC.recvLB, Lxy, MPI_FLOAT, rank-1-nprocx, ntag+5, comm, MPI_STATUS_IGNORE);}
    if ( px > 0 and py > 0)
          {MPI_Send(BC.sendLB, Lxy, MPI_FLOAT, rank-1-nprocx, ntag+6, comm);}
    if ( px < nprocx-1 and py < nprocy-1 )
          {MPI_Recv(BC.recvRT, Lxy, MPI_FLOAT, rank+1+nprocx, ntag+6, comm, MPI_STATUS_IGNORE);}


    if ( py < nprocy-1 and px > 0 )
          {MPI_Send(BC.sendLT, Lxy, MPI_FLOAT, rank+nprocx-1, ntag+7, comm);}
    if ( py>0 and px < nprocx-1 )
          {MPI_Recv(BC.recvRB, Lxy, MPI_FLOAT, rank-nprocx+1, ntag+7, comm, MPI_STATUS_IGNORE);}
    if ( py>0 and px < nprocx-1)
          {MPI_Send(BC.sendRB, Lxy, MPI_FLOAT, rank-nprocx+1, ntag+8, comm);}
    if ( py < nprocy -1 and px > 0)
          {MPI_Recv(BC.recvLT, Lxy, MPI_FLOAT, rank+nprocx-1, ntag+8, comm, MPI_STATUS_IGNORE);}

}


void commu_BC(MPI_Comm comm, BC_buffs BC, params_MPI pM, int nt, int hd, int fnx, int fny, float* v1, float* v2, float* v3, float* v4, float* v5){

      float *ptr[5] = {v1, v2, v3, v4, v5};


      int blocksize_2d = 128;  // seems reduce the block size makes it a little faster, but around 128 is okay.
      int max_len = max(fny,fnx);
      int num_block_2d = (hd*max_len*5+blocksize_2d-1)/blocksize_2d;
      //num_block_2d = (hd*(fnx+fny)+blocksize_2d-1)/blocksize_2d;
     collect2<<< num_block_2d, blocksize_2d>>>(ptr, BC, fnx, fny, 5, max_len);
     //collect<<< num_block_2d, blocksize_2d>>>(ptr, BC, fnx, fny);
     cudaDeviceSynchronize();
    // print2d(BC.sendR, fny, 1);
     //MPI_Barrier( comm );      
     exchange_BC(comm, BC, hd, fnx, fny, nt, pM.rank, pM.px, pM.py, pM.nprocx, pM.nprocy);
     //print2d(BC.recvR, fny-2, 1);
     //MPI_Barrier( comm );
     //distribute<<<num_block_2d, blocksize_2d >>>(ptr, BC, fnx, fny);
     distribute2<<<num_block_2d, blocksize_2d >>>(ptr, BC, fnx, fny, 5, max_len);
     cudaDeviceSynchronize();      
}


void calc_qois(int* cur_tip, int* alpha, int fnx, int fny, int kt, int num_grains, \
  float* tip_y, float* frac, float* y, int* aseq, int* ntip, int* extra_area, int* tip_final, int* total_area, int*, loss_area, int move_count){

     // cur_tip here inludes the halo
     bool contin_flag = true;

     while(contin_flag){
       // at this line
        *cur_tip += 1;
        int zeros = 0;
        for (int i=1; i<fnx-1;i++){
             int C = fnx*(*cur_tip) + i;
             //if (alpha[C]==0){printf("find liquid at %d at line %d\n", i, cur_tip);contin_flag=false;break;}
             if (alpha[C]==0) { zeros+=1;}
             if (zeros>ZERO) {contin_flag=false;break;}
        }
     }
     *cur_tip -=1;
     tip_y[kt] = y[*cur_tip];
     ntip[kt] = *cur_tip+move_count;
     printf("frame %d, ntip %d, tip %f\n", kt, ntip[kt], tip_y[kt]);

     for (int j = 1; j<fny-1; j++){ 
         for (int i=1; i<fnx-1;i++){
            int C = fnx*j + i;
              if (alpha[C]>0){ 
                tip_final[kt*num_grains+alpha[C]-1] = j+move_count; 
                total_area[kt*num_grains+alpha[C]-1]+=1+loss_area[alpha[C]-1];
                if (j > *cur_tip) {extra_area[kt*num_grains+alpha[C]-1]+=1; }}
         }
     }


}

void calc_frac( int* alpha, int fnx, int fny, int nts, int num_grains, float* tip_y, float* frac, float* y, int* aseq, int* ntip, int* left_coor){
     
     int* counts= (int*) malloc(num_grains* sizeof(int));

     for (int kt=0; kt<nts+1;kt++){
     memset(counts, 0, num_grains*sizeof(int));
     int cur_tip = ntip[kt];
     printf("cur_tip, %d\n",cur_tip);
       // pointer points at the first grid

     int summa=0;

     for (int i=1; i<fnx-1;i++){
            int C = fnx*cur_tip + i;
            if (alpha[C]>0){counts[alpha[C]-1]+=1;}
      }

     for (int j=0; j<num_grains;j++){

       frac[kt*num_grains+j] = counts[j]*1.0/(fnx-2);
       summa += counts[j];//frac[kt*num_grains+j];
       printf("grainID %d, counts %d, the current fraction: %f\n", j, counts[j], frac[kt*num_grains+j]);
     }
     if (summa<fnx-2-ZERO) {printf("the summation %d is off\n", summa);exit(1);}
     if ((summa<fnx-2) && (summa>=fnx-2-ZERO)){
        for (int grainj = 0; grainj<num_grains; grainj++) {frac[kt*num_grains+grainj]*= (fnx-2)*1.0f/summa; printf("grainID %d, the current fraction: %f\n", grainj, frac[kt*num_grains+grainj]);}
     }
     printf("summation %d\n", summa);     
     }
}

__global__ void 
move_frame(float* ph_buff, float* y_buff, float* ph, float* y, int* alpha, int* alpha_full, int* loss_area, int move_count, int fnx, int fny){

  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int PF_id = C/(fnx*fny);
  int pf_C = C - PF_id*fnx*fny;  // local C in every PF
  int j=pf_C/fnx; 
  int i=pf_C-j*fnx;

    if ( (i==0) && (j>0) && (j<fny-2) && (PF_id==0) ) {
        y_buff[j] = y[j+1];}

    if ( (i==0) &&  (j==fny-2) && (PF_id==0) ) {        
        y_buff[j] = 2*y[fny-2] - y[fny-3];}

    if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-2) && (PF_id<NUM_PF) ) {     
        ph_buff[C] = ph[C+fnx];}

    if ( (i>0) && (i<fnx-1) && (j==fny-2) && (PF_id<NUM_PF) ) {

        ph_buff[C] = 2*ph[C] - ph[C-fnx];}

    // add last layer of alpha to alpha_full[move_count]
    if ( (i<fnx) && (j==1) && (PF_id==0) ) {

        alpha_full[move_count*fnx+C] = alpha[C];
        atomicAdd(loss_area+alpha[C]-1,1);
        //printf("%d ", alpha[C]);
    }


}

__global__ void
copy_frame(float* ph_buff, float* y_buff, float* ph, float* y, int fnx, int fny){

  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int PF_id = C/(fnx*fny);
  int pf_C = C - PF_id*fnx*fny;  // local C in every PF
  int j=pf_C/fnx; 
  int i=pf_C-j*fnx;

  if ( (i == 0) && (j>0) && (j < fny-1) && (PF_id==0) ){
        y[j] = y_buff[j];}

  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) && (PF_id<NUM_PF) ) { 
        ph[C] = ph_buff[C];}


}


__global__ void
ave_x(float* phi, float* meanx, int fnx, int fny){


  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int PF_id = C/(fnx*fny);
  int pf_C = C - PF_id*fnx*fny;  // local C in every PF
  int j=pf_C/fnx; 
  int i=pf_C-j*fnx;
 
   if (C<fnx*fny*NUM_PF){
      atomicAdd(meanx+j,phi[C]);

   } 

}



void tip_mvf(int *cur_tip, float* phi, float* meanx, float* meanx_host, int fnx, int fny){

     int length = fnx*fny;
     int blocksize_2d = 128; 
     int num_block_PF = (length*NUM_PF+blocksize_2d-1)/blocksize_2d;

     ave_x<<<num_block_PF, blocksize_2d>>>(phi, meanx,fnx, fny);

     cudaMemcpy(meanx_host, meanx, fny * sizeof(float),cudaMemcpyDeviceToHost);
     while( (meanx_host[*cur_tip]/(NUM_PF*fnx)>LS) && (*cur_tip<fny-1) ) {*cur_tip+=1;}
    // for (int ww=0; ww<fny; ww++){ printf("avex %f \n",meanx_host[ww]/fnx);}
//      printf("currrent tip %d \n", *cur_tip);   
     cudaMemset(meanx,0,fny * sizeof(float));

}


void setup( params_MPI pM, GlobalConstants params, Mac_input mac, int fnx, int fny, int fny_f, float* x, float* y, float* phi, float* psi,float* U, int* alpha, \
  int* alpha_i_full, float* tip_y, float* frac, int* aseq, int* extra_area, int* tip_final, int* total_area){
  // we should have already pass all the data structure in by this time
  // move those data onto device
  int num_gpus_per_node = 4;
  int device_id_innode = pM.rank % num_gpus_per_node;
    //gpu_name = cuda.select_device( )
  cudaSetDevice(device_id_innode); 
  printCudaInfo(pM.rank,device_id_innode);
  
  float* x_device;// = NULL;
  float* y_device;// = NULL;
  float* y_device2;

  float* phi_old;// = NULL;
  float* phi_new;// = NULL;
  int* alpha_m;
  int* d_alpha_full;
  int* nucl_status;
  int* argmax;
  int* left_coor = (int*) malloc(params.num_theta* sizeof(int));
 
  for (int i=0; i<params.num_theta; i++){left_coor[i]=1;}
 
  float* PFs_old;
  float* PFs_new;
  // allocate x, y, phi, psi, U related params
  int cnx = fnx/(2*params.pts_cell+1);
  int cny = fny/(2*params.pts_cell+1);

  int length = fnx*fny;
  int length_c = cnx*cny;

  int move_count = 0;
  cudaMalloc((void **)&x_device, sizeof(float) * fnx);
  cudaMalloc((void **)&y_device, sizeof(float) * fny);
  cudaMalloc((void **)&y_device2, sizeof(float) * fny);

  cudaMalloc((void **)&phi_old,  sizeof(float) * length);
  cudaMalloc((void **)&phi_new,  sizeof(float) * length);
  cudaMalloc((void **)&alpha_m,    sizeof(int) * length);  
  cudaMalloc((void **)&d_alpha_full,   sizeof(int) * fnx*fny_f);  
  cudaMalloc((void **)&nucl_status,    sizeof(int) * length_c);
    cudaMalloc((void **)&PFs_old,    sizeof(float) * length * NUM_PF);
    cudaMalloc((void **)&PFs_new,    sizeof(float) * length * NUM_PF);
  cudaMalloc((void **)&argmax,    sizeof(int) * length);
  cudaMemset(argmax,0,sizeof(int) * length);

  cudaMemcpy(x_device, x, sizeof(float) * fnx, cudaMemcpyHostToDevice);
  cudaMemcpy(y_device, y, sizeof(float) * fny, cudaMemcpyHostToDevice);
  cudaMemcpy(y_device2, y, sizeof(float) * fny, cudaMemcpyHostToDevice);
  cudaMemcpy(phi_old, phi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(phi_new, phi, sizeof(float) * length, cudaMemcpyHostToDevice);

  cudaMemcpy(alpha_m, alpha, sizeof(int) * length, cudaMemcpyHostToDevice);

  // pass all the read-only params into global constant
  cudaMemcpyToSymbol(cP, &params, sizeof(GlobalConstants) );

   curandState* dStates;
   int period = params.noi_period;
   cudaMalloc((void **) &dStates, sizeof(curandState) * (length+period));
  // float* random_nums;
  // cudaMalloc((void **) &random_nums, sizeof(float) * (length+period));

  // MPI send/recv buffers
  BC_buffs SR_buffs;
  BC_buffs *buff_pointer = &SR_buffs; 
  allocate_mpi_buffs(params, buff_pointer, fnx, fny);
  //static int max_var = 5;

  //---macrodata for interpolation

  Mac_input Mgpu;
  cudaMalloc((void **)&(Mgpu.X_mac),  sizeof(float) * mac.Nx);
  cudaMalloc((void **)&(Mgpu.Y_mac),  sizeof(float) * mac.Ny);
  cudaMalloc((void **)&(Mgpu.t_mac),    sizeof(float) * mac.Nt);
  cudaMalloc((void **)&(Mgpu.T_3D),    sizeof(float) * mac.Nx*mac.Ny*mac.Nt);
  cudaMalloc((void **)&(Mgpu.theta_arr),    sizeof(float) * (NUM_PF+1) );
  cudaMalloc((void **)&(Mgpu.cost),    sizeof(float) * (NUM_PF+1) );
  cudaMalloc((void **)&(Mgpu.sint),    sizeof(float) * (NUM_PF+1) );
  cudaMemcpy(Mgpu.X_mac, mac.X_mac, sizeof(float) * mac.Nx, cudaMemcpyHostToDevice);  
  cudaMemcpy(Mgpu.Y_mac, mac.Y_mac, sizeof(float) * mac.Ny, cudaMemcpyHostToDevice); 
  cudaMemcpy(Mgpu.t_mac, mac.t_mac, sizeof(float) * mac.Nt, cudaMemcpyHostToDevice);  
  cudaMemcpy(Mgpu.T_3D, mac.T_3D, sizeof(float) * mac.Nt* mac.Nx* mac.Ny, cudaMemcpyHostToDevice);   
  cudaMemcpy(Mgpu.theta_arr, mac.theta_arr, sizeof(float) * (NUM_PF+1), cudaMemcpyHostToDevice);
  cudaMemcpy(Mgpu.cost, mac.cost, sizeof(float) * (NUM_PF+1), cudaMemcpyHostToDevice);
  cudaMemcpy(Mgpu.sint, mac.sint, sizeof(float) * (NUM_PF+1), cudaMemcpyHostToDevice);

//--

   int blocksize_1d = 128;
   int blocksize_2d = 128;  // seems reduce the block size makes it a little faster, but around 128 is okay.
   int num_block_2d = (fnx*fny+blocksize_2d-1)/blocksize_2d;
   int num_block_1d = (fnx+fny+blocksize_1d-1)/blocksize_1d;
   int num_block_PF = (length*NUM_PF+blocksize_2d-1)/blocksize_2d;
   int max_len = max(fny,fnx);
   int num_block_PF1d =  ( max_len*NUM_PF +blocksize_1d-1)/blocksize_1d;
   printf("block size %d, # blocks %d\n", blocksize_2d, num_block_2d); 

   int num_block_c = (cnx*cny+blocksize_2d-1)/blocksize_2d;   
   //init_nucl_status<<< num_block_c, blocksize_2d >>>(phi_old, nucl_status, cnx, cny, fnx); 
   //initialize<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, U_new, x_device, y_device, fnx, fny);
   //init_rand_num<<< (fnx*fny+period+blocksize_2d-1)/blocksize_2d, blocksize_2d >>>(dStates, params.seed_val,length+period);
  // gen_rand_num<<< (fnx*fny+period+blocksize_2d-1)/blocksize_2d,blocksize_2d >>>(dStates, random_nums,length+period);
   set_minus1<<< num_block_PF, blocksize_2d>>>(PFs_old,length*NUM_PF);
   set_minus1<<< num_block_PF, blocksize_2d>>>(PFs_new,length*NUM_PF);
   ini_PF<<< num_block_PF, blocksize_2d >>>(PFs_old, phi_old, alpha_m, length);
   ini_PF<<< num_block_PF, blocksize_2d >>>(PFs_new, phi_old, alpha_m, length);
   set_minus1<<< num_block_2d, blocksize_2d >>>(phi_old,length);

  // commu_BC(comm, SR_buffs, pM, params.Mt+2, params.ha_wd, fnx, fny, psi_new, phi_new, U_old, dpsi, alpha_m);
  // commu_BC(comm, SR_buffs, pM, params.Mt+3, params.ha_wd, fnx, fny, psi_old, phi_old, U_new, dpsi, alpha_m);
     //cudaDeviceSynchronize();
   set_BC_mpi_x<<< num_block_PF1d, blocksize_1d >>>(PFs_new, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
   set_BC_mpi_y<<< num_block_PF1d, blocksize_1d >>>(PFs_new, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
   set_BC_mpi_x<<< num_block_PF1d, blocksize_1d >>>(PFs_old, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
   set_BC_mpi_y<<< num_block_PF1d, blocksize_1d >>>(PFs_old, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
   rhs_psi<<< num_block_PF, blocksize_2d >>>(PFs_old, PFs_new, x_device, y_device, fnx, fny, 0,\
0, Mgpu.X_mac, Mgpu.Y_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nt, dStates, Mgpu.cost, Mgpu.sint );
   //print2d(phi_old,fnx,fny);
   float t_cur_step;
   int kts = params.Mt/params.nts;
   printf("kts %d, nts %d\n",kts, params.nts);
   int cur_tip=1;
   int tip_front = 1;
   int tip_thres = (int) ((1-BLANK)*fny);
   printf("max tip can go: %d\n", tip_thres); 
   float* meanx;
   cudaMalloc((void **)&meanx, sizeof(float) * fny);
   cudaMemset(meanx,0, sizeof(float) * fny);
  // printf(" ymax %f \n",y[fny-2] ); 
   float* meanx_host=(float*) malloc(fny* sizeof(float));

   int* ntip=(int*) malloc((params.nts+1)* sizeof(int));
   int* loss_area=(int*) malloc((params.num_theta)* sizeof(int));
   int* d_loss_area;
   cudaMalloc((void **)&d_loss_area, sizeof(int) * params.num_theta); 
   cudaMemset(d_loss_area,0,sizeof(int) * params.num_theta); 
   calc_qois(&cur_tip, alpha, fnx, fny, 0, params.num_theta, tip_y, frac, y, aseq, ntip, extra_area, tip_final, total_area, loss_area, move_count);
   cudaDeviceSynchronize();
   double startTime = CycleTimer::currentSeconds();
   for (int kt=0; kt<params.Mt/2; kt++){
   //for (int kt=0; kt<1; kt++){
  //   cudaDeviceSynchronize();
     //if ( (2*kt+2)%period==0) gen_rand_num<<< (fnx*fny+period+blocksize_2d-1)/blocksize_2d,blocksize_2d >>>(dStates, random_nums,length+period);

     //if ( params.ha_wd==1 ) commu_BC(comm, SR_buffs, pM, 2*kt, params.ha_wd, fnx, fny, psi_new, phi_new, U_old, dpsi, alpha_m);
     //cudaDeviceSynchronize();
     set_BC_mpi_x<<< num_block_PF1d, blocksize_1d >>>(PFs_new, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
     set_BC_mpi_y<<< num_block_PF1d, blocksize_1d >>>(PFs_new, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
     //cudaDeviceSynchronize();
   //  rhs_U<<< num_block_2d, blocksize_2d >>>(U_old, U_new, phi_new, dpsi, fnx, fny);


     //cudaDeviceSynchronize();
     t_cur_step = (2*kt+1)*params.dt*params.tau0;
     rhs_psi<<< num_block_PF, blocksize_2d >>>(PFs_new, PFs_old, x_device, y_device, fnx, fny, 2*kt+1,\
t_cur_step, Mgpu.X_mac, Mgpu.Y_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nt, dStates, Mgpu.cost, Mgpu.sint );
 
//     add_nucl<<<num_block_c, blocksize_2d>>>(nucl_status, cnx, cny, PFs_old, alpha_m, x_device, y_device, fnx, fny, dStates, \
        2.0f*params.dt*params.tau0, t_cur_step, Mgpu.X_mac, Mgpu.Y_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nt); 
     set_BC_mpi_x<<< num_block_PF1d, blocksize_1d >>>(PFs_old, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
     set_BC_mpi_y<<< num_block_PF1d, blocksize_1d >>>(PFs_old, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);

    
     if ( (2*kt+2)%kts==0) {
             //tip_mvf(&cur_tip,phi_new, meanx, meanx_host, fnx,fny);
             cudaMemset(alpha_m, 0, sizeof(int) * length);
             collect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, length, argmax);
             cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost); 
             cudaMemcpy(loss_area, d_loss_area, params.num_theta * sizeof(int),cudaMemcpyDeviceToHost);
             cudaMemcpy(y, y_device, fny * sizeof(int),cudaMemcpyDeviceToHost); 
             //QoIs based on alpha field
             cur_tip=0;
             calc_qois(&cur_tip, alpha, fnx, fny, (2*kt+2)/kts, params.num_theta, tip_y, frac, y, aseq,ntip,extra_area,tip_final,total_area, loss_area, move_count);
          }

     if ( (2*kt+2)%TIPP==0) {
             tip_mvf(&tip_front, PFs_old, meanx, meanx_host, fnx,fny);
             while (tip_front >=tip_thres){
                collect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, length, argmax);
                move_frame<<< num_block_PF, blocksize_2d >>>(PFs_new, y_device2, PFs_old, y_device, alpha_m, d_alpha_full, d_loss_area, move_count, fnx, fny);
                copy_frame<<< num_block_PF, blocksize_2d >>>(PFs_new, y_device2, PFs_old, y_device, fnx, fny);
                move_count +=1;
                tip_front-=1;
                //printf("moving count %d \n", move_count);
               // printf("current tip location %d, y %3.2f \n", cur_tip, y[cur_tip]);
   //cudaMemcpy(y, y_device2, fny * sizeof(float),cudaMemcpyDeviceToHost);
   //printf(" ymax %f \n",y[fny-3] );

             }
          //cudaMemcpy(y, y_device, fny * sizeof(int),cudaMemcpyDeviceToHost);
          //printf("current tip location %d, y %3.2f \n", cur_tip, y[cur_tip]);
          set_BC_mpi_x<<< num_block_PF1d, blocksize_1d >>>(PFs_old, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
          set_BC_mpi_y<<< num_block_PF1d, blocksize_1d >>>(PFs_old, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
          //if ((2*kt+2)%1000==0) printf("currrent tip %d \n", cur_tip);
          }
     
     //if ( (2*kt+2)%params.ha_wd==0 )commu_BC(comm, SR_buffs, pM, 2*kt+1, params.ha_wd, fnx, fny, psi_old, phi_old, U_new, dpsi, alpha_m);
     //cudaDeviceSynchronize();

     //cudaDeviceSynchronize();
   //  rhs_U<<< num_block_2d, blocksize_2d >>>(U_new, U_old, phi_old, dpsi, fnx, fny);
     //cudaDeviceSynchronize();*/
     t_cur_step = (2*kt+2)*params.dt*params.tau0;
     rhs_psi<<< num_block_PF, blocksize_2d >>>(PFs_old, PFs_new, x_device, y_device, fnx, fny, 2*kt+2,\
t_cur_step, Mgpu.X_mac, Mgpu.Y_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nt, dStates, Mgpu.cost, Mgpu.sint );


   }
   // params.kin_delta -= 0.05;
  // for (int i=0;i<params.nts+1;i++){
  //     printf("ntip %d \n", ntip[i]);
  // }


   cudaDeviceSynchronize();
   double endTime = CycleTimer::currentSeconds();
   printf("time for %d iterations: %f s\n", params.Mt, endTime-startTime);

   
   cudaMemset(alpha_m, 0, sizeof(int) * length);
   collect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, length, argmax); 
   cudaMemcpy(phi, phi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
   cudaMemcpy(alpha_i_full, d_alpha_full, fnx*fny_f * sizeof(int),cudaMemcpyDeviceToHost);
   cudaMemcpy(alpha_i_full+move_count*fnx, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
   calc_frac(alpha_i_full, fnx, fny, params.nts, params.num_theta, tip_y, frac, y, aseq, ntip, left_coor);

  cudaFree(x_device); cudaFree(y_device); cudaFree(y_device2);
  cudaFree(phi_old); cudaFree(phi_new);
  cudaFree(nucl_status); 
  cudaFree(dStates); //cudaFree(random_nums);
  cudaFree(PFs_new); cudaFree(PFs_old);
  cudaFree(alpha_m); cudaFree(argmax);
  cudaFree(nucl_status);
}







void printCudaInfo(int rank, int i)
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);


        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("rank %d, Device %d: %s\n", rank, i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);

    printf("---------------------------------------------------------\n"); 
}

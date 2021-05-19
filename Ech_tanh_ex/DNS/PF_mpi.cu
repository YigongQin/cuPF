#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <mpi.h>
#include "CycleTimer.h"

using namespace std;
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCKSIZE BLOCK_DIM_X*BLOCK_DIM_Y
#define HALO 1//halo in global region
#define REAL_DIM_X 14 //BLOCK_DIM_X-2*HALO
#define REAL_DIM_Y 14 //BLOCK_DIM_Y-2*HALO
#define LS -0.995
#define ACR 1e-5
void printCudaInfo(int rank, int i);
extern float toBW(int bytes, float sec);

struct GlobalConstants {
  int nx;
  int ny;
  int Mt;
  int nts; 
  int ictype;
  float G;
  float R;
  float delta;
  float k;
  float c_infm;
  float Dl;
  float d0;
  float W0;
  float lT;
  float lamd;
  float tau0;
  float c_infty;
  float R_tilde;
  float Dl_tilde;
  float lT_tilde;
  float eps;
  float alpha0;
  float dx;
  float dt;
  float asp_ratio;
  float lxd;
  float lx;
  float lyd;
  float eta;
  float U0;
  // parameters that are not in the input file
  float hi;
  float cosa;
  float sina;
  float sqrt2;
  float a_s;
  float epsilon;
  float a_12;
  float dt_sqrt;
  int noi_period;
  int seed_val;
  // MPI-related
  float Ti;
  int ha_wd;
  int Mnx;
  int Mny;
  int Mnt;
  float xmin;
  float ymin;

};

struct params_MPI{

    int rank;
    int px;
    int py;
    int nproc;
    int nprocx;
    int nprocy;
    int nx_loc;
    int ny_loc;
};

struct Mac_input{
  int Nx;
  int Ny;
  int Nt;
  float* X_mac; 
  float* Y_mac; 
  float* t_mac;
  float* alpha_mac;
  float* psi_mac;
  float* U_mac;
  float* T_3D;
};
  
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
set_BC_mpi_y(float* ps, float* ph, float* U, float* dpsi, float* ph2, int fnx, int fny, \
        int px, int py, int nprocx, int nprocy, int ha_wd){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
    
  if ( index<fnx ){
  // orignially it is 0,2; fny-1, fny-3
  // now ha_wd-1, ha_wd+1; fny-ha_wd, fny-ha_wd-2
      if (py==0) {
          int b_out = index+(ha_wd-1)*fnx;
          int b_in = index+(ha_wd+1)*fnx;

            ps[b_out] = ps[b_in];
            ph[b_out] = ph[b_in];
            U[b_out] = U[b_in];
            dpsi[b_out] = dpsi[b_in];
            ph2[b_out] = ph2[b_in];
            }
       if (py==nprocy-1) {     
          int t_out = index+(fny-ha_wd)*fnx;
          int t_in = index+(fny-ha_wd-2)*fnx;        
            ps[t_out] = ps[t_in];
            ph[t_out] = ph[t_in];
            U[t_out] = U[t_in];
            dpsi[t_out] = dpsi[t_in];
            ph2[t_out] = ph2[t_in];
           }
  }

}
__global__ void
set_BC_mpi_x(float* ps, float* ph, float* U, float* dpsi, float* ph2, int fnx, int fny, \
        int px, int py, int nprocx, int nprocy, int ha_wd){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if ( index<fny ){
  // orignially it is 0,2; fnx-1, fnx-3
  // now ha_wd-1, ha_wd+1; fnx-ha_wd, fnx-ha_wd-2  
      if (px==0) {
      
         int l_out = index*fnx + ha_wd-1;
         int l_in = index*fnx + ha_wd+1;
            ps[l_out] = ps[l_in];
            ph[l_out] = ph[l_in];
            U[l_out] = U[l_in];
            dpsi[l_out] = dpsi[l_in];
            ph2[l_out] = ph2[l_in];
      }
      if (px==nprocx-1){
      
         int r_out = index*fnx + fnx-ha_wd;
         int r_in = index*fnx + fnx-ha_wd-2;    
            ps[r_out] = ps[r_in];
            ph[r_out] = ph[r_in];
            U[r_out] = U[r_in];
            dpsi[r_out] = dpsi[r_in];
            ph2[r_out] = ph2[r_in];
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

// psi equation
__global__ void
rhs_psi(float* ps, float* ph, float* U, float* ps_new, float* ph_new, float* x, float* y, float* dpsi, int fnx, int fny, int nt, \
       float t, float* X, float* Y, float* Tmac, float* u_3d, int Nx, int Ny, int Nt, curandState* states, float* alpha_m ){

  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int j=C/fnx; 
  int i=C-j*fnx;
  // macros
   float Dt = Tmac[1]-Tmac[0];
   int kt = (int) ((t-Tmac[0])/Dt);
  // printf("%d ",kt);
   float delta_t = (t-Tmac[0])/Dt-kt;
   //printf("%f ",Dt);
   float Dx = X[1]-X[0]; // (X[Nx-1]-X[0]) / (Nx-1)
   float Dy = Y[1]-Y[0];
  // if the points are at boundary, return
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) ) {
       // find the indices of the 8 neighbors for center
       //if (C==1000){printf("find");}
       int R=C+1;
       int L=C-1;
       int T=C+fnx;
       int B=C-fnx;
       float alpha = alpha_m[C];
       float sina = sinf(alpha);
       float cosa = cosf(alpha);
        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================

        // these ps's are defined on cell centers
        float psipjp=( ps[C] + ps[R] + ps[T] + ps[T+1] ) * 0.25f;
        float psipjm=( ps[C] + ps[R] + ps[B] + ps[B+1] ) * 0.25f;
        float psimjp=( ps[C] + ps[L] + ps[T-1] + ps[T] ) * 0.25f;
        float psimjm=( ps[C] + ps[L] + ps[B-1] + ps[B] ) * 0.25f;

        float phipjp=( ph[C] + ph[R] + ph[T] + ph[T+1] ) * 0.25f;
        float phipjm=( ph[C] + ph[R] + ph[B] + ph[B+1] ) * 0.25f;
        float phimjp=( ph[C] + ph[L] + ph[T-1] + ph[T] ) * 0.25f;
        float phimjm=( ph[C] + ph[L] + ph[B-1] + ph[B] ) * 0.25f;
        
        // ============================
        // right edge flux
        // ============================
        float psx = ps[R]-ps[C];
        float psz = psipjp - psipjm;
        float phx = ph[R]-ph[C];
        float phz = phipjp - phipjm;

        float A  = atheta( phx,phz,cosa,sina);
        float Ap = aptheta(phx,phz,cosa,sina);
        float JR = A * ( A*psx - Ap*psz );
        
        // ============================
        // left edge flux
        // ============================
        psx = ps[C]-ps[L];
        psz = psimjp - psimjm;
        phx = ph[C]-ph[L];
        phz = phimjp - phimjm; 

        A  = atheta( phx,phz,cosa,sina);
        Ap = aptheta(phx,phz,cosa,sina);
        float JL = A * ( A*psx - Ap*psz );
        
        // ============================
        // top edge flux
        // ============================
        psx = psipjp - psimjp;
        psz = ps[T]-ps[C];
        phx = phipjp - phimjp;
        phz = ph[T]-ph[C];

        A  = atheta( phx,phz,cosa,sina);
        Ap = aptheta(phx,phz,cosa,sina);
        float JT = A * ( A*psz + Ap*psx );

        // ============================
        // bottom edge flux
        // ============================
        psx = psipjm - psimjm;
        psz = ps[C]-ps[B];
        phx = phipjm - phimjm;
        phz = ph[C]-ph[B];

        A  = atheta( phx,phz,cosa,sina);
        Ap = aptheta(phx,phz,cosa,sina);
        float JB = A * ( A*psz + Ap*psx );

         /*# =============================================================
        #
        # 2. EXTRA TERM: sqrt2 * atheta**2 * phi * |grad psi|^2
        #
        # =============================================================
        # d(phi)/dx  d(psi)/dx d(phi)/dz  d(psi)/dz at nodes (i,j)*/
        float phxn = ( ph[R] - ph[L] ) * 0.5f;
        float phzn = ( ph[T] - ph[B] ) * 0.5f;
        float psxn = ( ps[R] - ps[L] ) * 0.5f;
        float pszn = ( ps[T] - ps[B] ) * 0.5f;

        float A2 = atheta(phxn,phzn,cosa,sina);
        A2 = A2*A2;
        float gradps2 = (psxn)*(psxn) + (pszn)*(pszn);
        float extra =  -cP.sqrt2 * A2 * ph[C] * gradps2;

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
      int offset =  kx + ky*Nx + kt*Nx*Ny;
      int offset_n =  kx + ky*Nx + (kt+1)*Nx*Ny;
     // printf("%d ", Nx);
      float Tinterp= ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset ] + (1.0f-delta_x)*delta_y*u_3d[ offset+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset+1 ] +   delta_x*delta_y*u_3d[ offset+Nx+1 ] )*(1.0f-delta_t) + \
             ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset_n ] + (1.0f-delta_x)*delta_y*u_3d[ offset_n+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset_n+1 ] +   delta_x*delta_y*u_3d[ offset_n+Nx+1 ] )*delta_t;

        float Up = (Tinterp-cP.Ti)/(cP.c_infm/cP.k)/(1.0-cP.k);  //(y[j]/cP.W0 - cP.R_tilde * (nt*cP.dt) )/cP.lT_tilde;

        float rhs_psi = ((JR-JL) + (JT-JB) + extra) * cP.hi*cP.hi + \
                   cP.sqrt2*ph[C] - cP.lamd*(1.0f-ph[C]*ph[C])*cP.sqrt2*(U[C] + Up);

        /*# =============================================================
        #
        # 4. dpsi/dt term
        #
        # =============================================================*/
        float tp = (1.0f-(1.0f-cP.k)*Up);
        float tau_psi;
        if (tp >= cP.k){tau_psi = tp*A2;}
               else {tau_psi = cP.k*A2;}
        
        dpsi[C] = rhs_psi / tau_psi; 
        
        float rand;
        if ( ( ph[C]>-0.995 ) && ( ph[C]<0.995 ) ) {rand= cP.dt_sqrt*cP.hi*cP.eta*(curand_uniform(states+C)-0.5);}
        else {rand = 0.0f;}      //  ps_new[C] = ps[C] +  cP.dt * dpsi[C];
        //int new_noi_loc = nt%cP.noi_period;//*cP.seed_val)%(fnx*fny);
        ps_new[C] = ps[C] +  cP.dt * dpsi[C] + rand; //cP.dt_sqrt*cP.hi*cP.eta*rnd[C+new_noi_loc];
        ph_new[C] = tanhf(ps_new[C]/cP.sqrt2);

        if ( (ph_new[C] > LS) && (ph[C] < LS) ){
           if ( (alpha<ACR) && (alpha>-ACR) ){
            int n1 = 0; int n2=0; int n3=0;
            float alpha1 = 0.0f; float alpha2 = 0.0f; float alpha3 = 0.0f;
              for (int locj=-1;locj<2;locj++){
                  for (int loci=-1;loci<2;loci++){
                     int os_loc = locj*fnx+loci+C;
                     if (ph[os_loc]>LS){
                       float atemp = alpha_m[os_loc];
                       if      (alpha1 == 0.0f){n1+=1; alpha1=atemp;}
                       else if ( (atemp-alpha1<ACR) && (atemp-alpha1>-ACR) ){n1+=1;}
                       else if (alpha2 == 0.0f){n2+=1; alpha2=atemp;}
                       else if ( (atemp-alpha2<ACR) && (atemp-alpha2>-ACR) ){n2+=1;}
                       else if (alpha3 == 0.0f){n3+=1; alpha3=atemp;}
                       else if ( (atemp-alpha3<ACR) && (atemp-alpha3>-ACR) ){n3+=1;}
                       else{printf("case not closed!!!\n");}
                     }  
             }}
           if ((n1>=n2) && (n1>=n3)) {alpha_m[C]=alpha1;}
           else if (n2>=n3) {alpha_m[C]=alpha2;}
           else {alpha_m[C]=alpha3;} 
          }
       }
        //if (C==1000){printf("%f ",ph_new[C]);}
     }
} 

// U equation
__global__ void
rhs_U(float* U, float* U_new, float* ph, float* dpsi, int fnx, int fny ){

  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int j=C/fnx;
  int i=C-j*fnx;
  // if the points are at boundary, return
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) ) {
        // find the indices of the 8 neighbors for center
        int R=C+1;
        int L=C-1;
        int T=C+fnx;
        int B=C-fnx;
        float hi = cP.hi;
        float Dl_tilde = cP.Dl_tilde;
        float k = cP.k;
        float nx,nz;
        float eps = cP.eps;
        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================

        // these ps's are defined on cell centers
        float phipjp=( ph[C] + ph[R] + ph[T] + ph[T+1] ) * 0.25f;
        float phipjm=( ph[C] + ph[R] + ph[B] + ph[B+1] ) * 0.25f;
        float phimjp=( ph[C] + ph[L] + ph[T-1] + ph[T] ) * 0.25f;
        float phimjm=( ph[C] + ph[L] + ph[B-1] + ph[B] ) * 0.25f;

        float jat    = 0.5f*(1.0f+(1.0f-k)*U[C])*(1.0f-ph[C]*ph[C])*dpsi[C];
        /*# ============================
        # right edge flux (i+1/2, j)
        # ============================*/
        float phx = ph[R]-ph[C];
        float phz = phipjp - phipjm;
        float phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nx = phx / sqrtf(phn2);}
                   else {nx = 0.0f;}
        
        float jat_ip = 0.5f*(1.0f+(1.0f-k)*U[R])*(1.0f-ph[R]*ph[R])*dpsi[R];	
        float UR = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[R])*(U[R]-U[C]) + 0.5f*(jat + jat_ip)*nx;
    	 
    	 
        /* ============================
        # left edge flux (i-1/2, j)
        # ============================*/
        phx = ph[C]-ph[L];
        phz = phimjp - phimjm;
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nx = phx / sqrtf(phn2);}
                   else {nx = 0.0f;}
        
        float jat_im = 0.5f*(1.0f+(1.0f-k)*U[L])*(1.0f-ph[L]*ph[L])*dpsi[L];
        float UL = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[L])*(U[C]-U[L]) + 0.5f*(jat + jat_im)*nx;
    	 
    	 
        /*# ============================
        # top edge flux (i, j+1/2)
        # ============================*/     
        phx = phipjp - phimjp;
        phz = ph[T]-ph[C];
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nz = phz / sqrtf(phn2);}
                   else {nz = 0.0f;}    	
  
        float jat_jp = 0.5f*(1.0f+(1.0f-k)*U[T])*(1.0f-ph[T]*ph[T])*dpsi[T];      
        
        float UT = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[T])*(U[T]-U[C]) + 0.5f*(jat + jat_jp)*nz;
    	 
    	 
        /*# ============================
        # bottom edge flux (i, j-1/2)
        # ============================*/  
        phx = phipjm - phimjm;
        phz = ph[C]-ph[B];
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nz = phz / sqrtf(phn2);}
                   else {nz = 0.0f;} 

        float jat_jm = 0.5f*(1.0f+(1.0f-k)*U[B])*(1.0f-ph[B]*ph[B])*dpsi[B];              
        float UB = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[B])*(U[C]-U[B]) + 0.5f*(jat + jat_jm)*nz;
        
        float rhs_U = ( (UR-UL) + (UT-UB) ) * hi + cP.sqrt2 * jat;
        float tau_U = (1.0f+cP.k) - (1.0f-cP.k)*ph[C];

        U_new[C] = U[C] + cP.dt * ( rhs_U / tau_U );

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


void setup(MPI_Comm comm,  params_MPI pM, GlobalConstants params, Mac_input mac, int fnx, int fny, float* x, float* y, float* phi, float* psi,float* U, float* alpha){
  // we should have already pass all the data structure in by this time
  // move those data onto device
  int num_gpus_per_node = 4;
  int device_id_innode = pM.rank % num_gpus_per_node;
    //gpu_name = cuda.select_device( )
  cudaSetDevice(device_id_innode); 
  printCudaInfo(pM.rank,device_id_innode);
  
  float* x_device;// = NULL;
  float* y_device;// = NULL;

  float* psi_old;// = NULL;
  float* psi_new;// = NULL;
  float* U_old;// = NULL;
  float* U_new;// = NULL;
  float* phi_old;// = NULL;
  float* phi_new;// = NULL;
  float* dpsi;// = NULL;
  float* alpha_m;
  // allocate x, y, phi, psi, U related params
  int length = fnx*fny;

  cudaMalloc((void **)&x_device, sizeof(float) * fnx);
  cudaMalloc((void **)&y_device, sizeof(float) * fny);

  cudaMalloc((void **)&phi_old,  sizeof(float) * length);
  cudaMalloc((void **)&psi_old,  sizeof(float) * length);
  cudaMalloc((void **)&U_old,    sizeof(float) * length);
  cudaMalloc((void **)&phi_new,  sizeof(float) * length);
  cudaMalloc((void **)&psi_new,  sizeof(float) * length);
  cudaMalloc((void **)&U_new,    sizeof(float) * length);
  cudaMalloc((void **)&dpsi,    sizeof(float) * length);
  cudaMalloc((void **)&alpha_m,    sizeof(float) * length);  

  cudaMemcpy(x_device, x, sizeof(float) * fnx, cudaMemcpyHostToDevice);
  cudaMemcpy(y_device, y, sizeof(float) * fny, cudaMemcpyHostToDevice);
  cudaMemcpy(psi_old, psi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(phi_old, phi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(U_old, U, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(psi_new, psi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(phi_new, phi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(U_new, U, sizeof(float) * length, cudaMemcpyHostToDevice);

  cudaMemcpy(alpha_m, alpha, sizeof(float) * length, cudaMemcpyHostToDevice);
  // pass all the read-only params into global constant
  cudaMemcpyToSymbol(cP, &params, sizeof(GlobalConstants) );

   curandState* dStates;
   int period = params.noi_period;
   cudaMalloc((void **) &dStates, sizeof(curandState) * (length+period));
   float* random_nums;
   cudaMalloc((void **) &random_nums, sizeof(float) * (length+period));

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
  cudaMemcpy(Mgpu.X_mac, mac.X_mac, sizeof(float) * mac.Nx, cudaMemcpyHostToDevice);  
  cudaMemcpy(Mgpu.Y_mac, mac.Y_mac, sizeof(float) * mac.Ny, cudaMemcpyHostToDevice); 
  cudaMemcpy(Mgpu.t_mac, mac.t_mac, sizeof(float) * mac.Nt, cudaMemcpyHostToDevice);  
  cudaMemcpy(Mgpu.T_3D, mac.T_3D, sizeof(float) * mac.Nt* mac.Nx* mac.Ny, cudaMemcpyHostToDevice);   



//--

   int blocksize_1d = 128;
   int blocksize_2d = 128;  // seems reduce the block size makes it a little faster, but around 128 is okay.
   int num_block_2d = (fnx*fny+blocksize_2d-1)/blocksize_2d;
   int num_block_1d = (fnx+fny+blocksize_1d-1)/blocksize_1d;
   printf("block size %d, # blocks %d\n", blocksize_2d, num_block_2d); 
   
   
   //initialize<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, U_new, x_device, y_device, fnx, fny);
   init_rand_num<<< (fnx*fny+period+blocksize_2d-1)/blocksize_2d, blocksize_2d >>>(dStates, params.seed_val,length+period);
   gen_rand_num<<< (fnx*fny+period+blocksize_2d-1)/blocksize_2d,blocksize_2d >>>(dStates, random_nums,length+period);

   commu_BC(comm, SR_buffs, pM, params.Mt+2, params.ha_wd, fnx, fny, psi_new, phi_new, U_old, dpsi, alpha_m);
   commu_BC(comm, SR_buffs, pM, params.Mt+3, params.ha_wd, fnx, fny, psi_old, phi_old, U_new, dpsi, alpha_m);
     //cudaDeviceSynchronize();
   set_BC_mpi_x<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_old, dpsi, phi_old, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
   set_BC_mpi_y<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_old, dpsi, phi_old, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
   set_BC_mpi_x<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_new, dpsi, phi_new, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
   set_BC_mpi_y<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_new, dpsi, phi_new, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
   rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, x_device, y_device, dpsi, fnx, fny, 0,\
0, Mgpu.X_mac, Mgpu.Y_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nt, dStates, alpha_m );
   //print2d(phi_old,fnx,fny);
   float t_cur_step;
   cudaDeviceSynchronize();
   double startTime = CycleTimer::currentSeconds();
   for (int kt=0; kt<params.Mt/2; kt++){
  //for (int kt=0; kt<0; kt++){
  //   cudaDeviceSynchronize();
     //if ( (2*kt+2)%period==0) gen_rand_num<<< (fnx*fny+period+blocksize_2d-1)/blocksize_2d,blocksize_2d >>>(dStates, random_nums,length+period);

     if ( params.ha_wd==1 ) commu_BC(comm, SR_buffs, pM, 2*kt, params.ha_wd, fnx, fny, psi_new, phi_new, U_old, dpsi, alpha_m);
     //cudaDeviceSynchronize();
     set_BC_mpi_x<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_old, dpsi, phi_old, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
     set_BC_mpi_y<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_old, dpsi, phi_old, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
     //cudaDeviceSynchronize();
     rhs_U<<< num_block_2d, blocksize_2d >>>(U_old, U_new, phi_new, dpsi, fnx, fny);


     //cudaDeviceSynchronize();
     t_cur_step = (2*kt+1)*params.dt*params.tau0;
     rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_new, phi_new, U_new, psi_old, phi_old, x_device, y_device, dpsi, fnx, fny, 2*kt+1,\
t_cur_step, Mgpu.X_mac, Mgpu.Y_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nt, dStates, alpha_m );
 
 
     
     if ( (2*kt+2)%params.ha_wd==0 )commu_BC(comm, SR_buffs, pM, 2*kt+1, params.ha_wd, fnx, fny, psi_old, phi_old, U_new, dpsi, alpha_m);
     //cudaDeviceSynchronize();
     set_BC_mpi_x<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_new, dpsi, phi_new, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
     set_BC_mpi_y<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_new, dpsi, phi_new, fnx, fny, pM.px, pM.py, pM.nprocx, pM.nprocy, params.ha_wd);
     //cudaDeviceSynchronize();
     rhs_U<<< num_block_2d, blocksize_2d >>>(U_new, U_old, phi_old, dpsi, fnx, fny);
     //cudaDeviceSynchronize();*/
     t_cur_step = (2*kt+2)*params.dt*params.tau0;
     rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, x_device, y_device, dpsi, fnx, fny, 2*kt+2,\
t_cur_step, Mgpu.X_mac, Mgpu.Y_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nt, dStates, alpha_m );

   }
   cudaDeviceSynchronize();
   double endTime = CycleTimer::currentSeconds();
   printf("time for %d iterations: %f s\n", params.Mt, endTime-startTime);
   cudaMemcpy(psi, psi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(phi, phi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(U, U_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(alpha, alpha_m, length * sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree(x_device); cudaFree(y_device);
  cudaFree(psi_old); cudaFree(psi_new);
  cudaFree(phi_old); cudaFree(phi_new);
  cudaFree(U_old); cudaFree(U_new);
  cudaFree(dpsi);  
  cudaFree(dStates); cudaFree(random_nums);
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

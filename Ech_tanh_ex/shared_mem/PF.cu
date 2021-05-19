#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include "CycleTimer.h"

using namespace std;

// this is dependent on the time tiling and grid size of one thread block
// we first finish a non-time tiling version

#define HALO 1//halo in global region
#define BLOCK_DIM_X 128
#define REAL_DIM 126 //BLOCK_DIM_X-2*HALO
#define SHARESIZE 384

void printCudaInfo();
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

};

__constant__ GlobalConstants cP;

// Device codes 

// boundary condition
// only use this function to access the boundary points, 
// other functions return at the boundary
// TODO: this function is doing what, we can definetly merge this into kenrel right?
__global__ void
set_BC(float* ps, float* ph, float* U, float* dpsi, int fnx, int fny){

  // find the location of boundary:
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // z=0, lx
  if (index<fnx) {
    int b_in = index+2*fnx;
    int t_out = index+(fny-1)*fnx;
    int t_in = index+(fny-3)*fnx;

    ps[index] = ps[b_in];
    ph[index] = ph[b_in];
    U[index] = U[b_in];
    dpsi[index] = dpsi[b_in];

    ps[t_out] = ps[t_in];
    ph[t_out] = ph[t_in];
    U[t_out] = U[t_in];
    dpsi[t_out] = dpsi[t_in];
  }
  if (index<fny){
    int l_out = index*fnx;
    int l_in = index*fnx + 2;
    int r_out = index*fnx + fnx -1;
    int r_in = index*fnx + fnx -3;
 
    ps[l_out] = ps[l_in];
    ph[l_out] = ph[l_in];
    U[l_out] = U[l_in];
    dpsi[l_out] = dpsi[l_in];
 
    ps[r_out] = ps[r_in];
    ph[r_out] = ph[r_in];
    U[r_out] = U[r_in];
    dpsi[r_out] = dpsi[r_in];
  }
}

// initialization
__global__ void
initialize(float* ps_old, float* ph_old, float* U_old, float* ps_new, float* ph_new, float* U_new
           , float* x, float* y, int fnx, int fny){

  int C = blockIdx.x * blockDim.x + threadIdx.x;
  // obtain i and j(2D position)
  int j=C/fnx;
  int i=C-j*fnx;
  // when initialize, you need to consider C/F layout
  // if F layout, the 1D array has peroidicity of nx    
  // all the variables should be functions of x and y
  // size (nx+2)*(ny+2), x:nx, y:ny
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) ) {
    float xc = x[i];
    float yc = y[j];
    int cent = fnx/2;
    ps_old[C] = 5.625f - sqrtf( (xc-x[cent])*(xc-x[cent]) + yc*yc )/cP.W0 ;
    //if (C<1000){printf("ps %f\n",ps_old[C]);}
    ps_new[C] = ps_old[C];
    U_old[C] = cP.U0;
    U_new[C] = cP.U0;
    ph_old[C] = tanhf(ps_old[C]/cP.sqrt2);
    ph_new[C] = tanhf(ps_new[C]/cP.sqrt2);
  //  if (C<1000){printf("phi %f\n",ph_old[C]);} 
  }
}

// anisotropy functions
__device__ float
atheta(float ux, float uz){
  
   float ux2 = cP.cosa*ux + cP.sina*uz;
         ux2 = ux2*ux2;
   float uz2 = -cP.sina*ux + cP.cosa*uz;
         uz2 = uz2*uz2;
   float MAG_sq = (ux2 + uz2);
   float MAG_sq2= MAG_sq*MAG_sq;
   if (MAG_sq > cP.eps){
         return cP.a_s*( 1.0f + cP.epsilon*(ux2*ux2 + uz2*uz2) / MAG_sq2);}
   else {return 1.0f;}
}


__device__ float
aptheta(float ux, float uz){

   float uxr = cP.cosa*ux + cP.sina*uz;
   float ux2 = uxr*uxr;
   float uzr = -cP.sina*ux + cP.cosa*uz;
   float uz2 = uzr*uzr;
   float MAG_sq = (ux2 + uz2);
   float MAG_sq2= MAG_sq*MAG_sq;
   if (MAG_sq > cP.eps){
         return -cP.a_12*uxr*uzr*(ux2 - uz2) / MAG_sq2;}
   else {return 0.0f;}
}

// psi & phi equation: two dimensions



// psi equation
__global__ void
rhs_psi(float* ps, float* ph, float* U, float* ps_new, float* ph_new, \
        float* y, float* dpsi, int fnx, int fny, int nt ){

  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int j=C/fnx; 
  int i=C-j*fnx;
  // if the points are at boundary, return
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) ) {
       // find the indices of the 8 neighbors for center
       //if (C==1000){printf("find");}
       int R=C+1;
       int L=C-1;
       int T=C+fnx;
       int B=C-fnx;
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

        // if (C==1001){
        //   printf("detailed check of neighbours 2\n");
        //   printf("R: %f ; L:%f ; T: %f ; B: %f \n", psipjp, psipjm, psimjp, psimjm);
        // }
        
        // ============================
        // right edge flux
        // ============================
        float psx = ps[R]-ps[C];
        float psz = psipjp - psipjm;
        float phx = ph[R]-ph[C];
        float phz = phipjp - phipjm;

        float A  = atheta( phx,phz);
        float Ap = aptheta(phx,phz);
        float JR = A * ( A*psx - Ap*psz );
        
        // ============================
        // left edge flux
        // ============================
        psx = ps[C]-ps[L];
        psz = psimjp - psimjm;
        phx = ph[C]-ph[L];
        phz = phimjp - phimjm; 

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        float JL = A * ( A*psx - Ap*psz );
        
        // ============================
        // top edge flux
        // ============================
        psx = psipjp - psimjp;
        psz = ps[T]-ps[C];
        phx = phipjp - phimjp;
        phz = ph[T]-ph[C];

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        float JT = A * ( A*psz + Ap*psx );

        // ============================
        // bottom edge flux
        // ============================
        psx = psipjm - psimjm;
        psz = ps[C]-ps[B];
        phx = phipjm - phimjm;
        phz = ph[C]-ph[B];

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
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

        float A2 = atheta(phxn,phzn);
        A2 = A2*A2;
        float gradps2 = (psxn)*(psxn) + (pszn)*(pszn);
        float extra =  -cP.sqrt2 * A2 * ph[C] * gradps2;

        /*# =============================================================
        #
        # 3. double well (transformed): sqrt2 * phi + nonlinear terms
        #
        # =============================================================*/

        float Up = (y[j]/cP.W0 - cP.R_tilde * (nt*cP.dt) )/cP.lT_tilde;

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
        
        ps_new[C] = ps[C] +  cP.dt * dpsi[C];
        ph_new[C] = tanhf(ps_new[C]/cP.sqrt2);
        // if (C == 1001){
        //   printf("check data ps: %f and ph: %f and dpsi: %f\n", ps_new[C], ph_new[C], dpsi[C]);
        // }
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

// (phi_new , psi_new, U_old, dpsi) with rotten BC  to (phi_old, psi_old, U_new) with rotten BC
__global__ void
merge_PF(float* ps, float* ph, float* U, float* ps_new, float* ph_new, float* U_new, float* dpsi, float* dpsi_new,\
       float* y, int fnx, int fny, int nt ){
  

  // load old data
  __shared__ float ps_shared[SHARESIZE];
  __shared__ float ph_shared[SHARESIZE];
  __shared__ float U_shared[SHARESIZE];
  __shared__ float dpsi_shared[SHARESIZE];
  // write data into new array and update at last
  __shared__ float ps_shared_new[SHARESIZE];
  __shared__ float ph_shared_new[SHARESIZE];
  __shared__ float U_shared_new[SHARESIZE];
  __shared__ float dpsi_shared_new[SHARESIZE];
  
  // local id in thread block
  int place = threadIdx.x; //0, BLOCK_DIM_X
  
  // block id in core region
  int block_id = blockIdx.x; // 0~num_block_x*num_block_y 


  int block_addr = block_id * REAL_DIM;


  // location in global addr (i, j)
  // add HALO due to global halo region; then minus the halo region in the thread block; last add the local id
  int C = block_addr + HALO - HALO + fnx+ place; // the according location of the global memory note here you
  // can reach the data only between 0<j<fny-1 
  int j = C/fnx ; 
  int i = C - fnx*j;
  
  // load necessary data
  if ((i < fnx) && (j < fny-1) && (j>0) ){
  ps_shared[place] = ps[C];
  ph_shared[place] = ph[C];
  U_shared[place]  = U[C];
  dpsi_shared[place]  = dpsi[C];
  
  ps_shared[place+ BLOCK_DIM_X] = ps[C+fnx];
  ph_shared[place+ BLOCK_DIM_X] = ph[C+fnx];
  U_shared[place+ BLOCK_DIM_X]  = U[C+fnx];
  dpsi_shared[place+ BLOCK_DIM_X]  = dpsi[C+fnx];
  
  ps_shared[place -BLOCK_DIM_X] = ps[C-fnx];
  ph_shared[place-BLOCK_DIM_X] = ph[C-fnx];
  U_shared[place-BLOCK_DIM_X]  = U[C-fnx];
  dpsi_shared[place-BLOCK_DIM_X]  = dpsi[C-fnx];  
  }
  __syncthreads();
  // updaate BC first
  // bottom line
  /*
  if ((j == 2) && (i < fnx) &&(place>=2*BLOCK_DIM_X) ){
      ps_shared[place - 2*BLOCK_DIM_X] = ps_shared[place];
      ph_shared[place - 2*BLOCK_DIM_X] = ph_shared[place];
      dpsi_shared[place - 2*BLOCK_DIM_X] = dpsi_shared[place];
      U_shared[place - 2*BLOCK_DIM_X] = U_shared[place];

      ps[C-2*fnx] = ps_shared[place];
      ph[C-2*fnx] = ph_shared[place];
      dpsi[C-2*fnx] = dpsi_shared[place];
      U[C-2*fnx] = U_shared[place];
  }
  // up line
  if ((j == fny - 3)&& (i < fnx) &&(place<BLOCK_DIM_X)){
      ps_shared[place+2*BLOCK_DIM_X] = ps_shared[place];
      ph_shared[place+2*BLOCK_DIM_X] = ph_shared[place];
      dpsi_shared[place+2*BLOCK_DIM_X]   = dpsi_shared[place];
      U_shared[place+2*BLOCK_DIM_X] = U_shared[place];
      
      ps[C+2*fnx] = ps_shared[place];
      ph[C+2*fnx] = ph_shared[place];
      dpsi[C+2*fnx] = dpsi_shared[place];
      U[C+2*fnx] = U_shared[place];
  }
   
  __syncthreads();
  
  // left line
  if ((i == 0) && (j < fny)){
      ps_shared[place] = ps_shared[place + 2];
      ph_shared[place] = ph_shared[place + 2];
      dpsi_shared[place]   = dpsi_shared[place + 2];
      U_shared[place] = U_shared[place + 2];

      ps[C] = ps_shared[place + 2];
      ph[C] = ph_shared[place + 2];
      dpsi[C] = dpsi_shared[place + 2];
      U[C] = U_shared[place + 2];
  }
  // right line
  if ((i == fnx - 1) && (j < fny)){
      ps_shared[place] = ps_shared[place - 2];
      ph_shared[place] = ph_shared[place - 2];
      dpsi_shared[place]   = dpsi_shared[place - 2];
      U_shared[place] = U_shared[place - 2];

      ps[C] = ps_shared[place - 2];
      ph[C] = ph_shared[place - 2];
      dpsi[C] = dpsi_shared[place - 2];
      U[C] = U_shared[place - 2];
  }
  
  __syncthreads();
  */
  // update U
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) ) {
    // only update the inner res
    if (0<place<BLOCK_DIM_X -1) {
      // find the indices of the 8 neighbors for center
        int R=place+1;
        int L=place-1;
        int T=place+BLOCK_DIM_X;
        int B=place-BLOCK_DIM_X;
        float hi = cP.hi;
        float Dl_tilde = cP.Dl_tilde;
        float k = cP.k;
        float nx, nz;
        float eps = cP.eps;
        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================

        // these ph's are defined on cell centers
        float phipjp=( ph_shared[place] + ph_shared[R] + ph_shared[T] + ph_shared[T+1] ) * 0.25f;
        float phipjm=( ph_shared[place] + ph_shared[R] + ph_shared[B] + ph_shared[B+1] ) * 0.25f;
        float phimjp=( ph_shared[place] + ph_shared[L] + ph_shared[T-1] + ph_shared[T] ) * 0.25f;
        float phimjm=( ph_shared[place] + ph_shared[L] + ph_shared[B-1] + ph_shared[B] ) * 0.25f;

        // if (C==1001){
        //   printf("detailed check of neighbours 3\n");
        //   printf("R: %f ; L:%f ; T: %f ; B: %f \n", phipjp, phipjm, phimjp, phimjm);
        // }
        float jat    = 0.5f*(1.0f+(1.0f-k)*U_shared[place])*(1.0f-ph_shared[place]*ph_shared[place])*dpsi_shared[place];
        /*# ============================
        # right edge flux (i+1/2, j)
        # ============================*/
        float phx = ph_shared[R]-ph_shared[place];
        float phz = phipjp - phipjm;
        float phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nx = phx / sqrtf(phn2);}
                   else {nx = 0.0f;}
        
        float jat_ip = 0.5f*(1.0f+(1.0f-k)*U_shared[R])*(1.0f-ph_shared[R]*ph_shared[R])*dpsi_shared[R];	
        float UR = hi*Dl_tilde*0.5f*(2.0f - ph_shared[place] - ph_shared[R])*(U_shared[R]-U_shared[place]) + 0.5f*(jat + jat_ip)*nx;
    	 
    	 
        /* ============================
        # left edge flux (i-1/2, j)
        # ============================*/
        phx = ph_shared[place]-ph_shared[L];
        phz = phimjp - phimjm;
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nx = phx / sqrtf(phn2);}
                   else {nx = 0.0f;}
        
        float jat_im = 0.5f*(1.0f+(1.0f-k)*U_shared[L])*(1.0f-ph_shared[L]*ph_shared[L])*dpsi_shared[L];
        float UL = hi*Dl_tilde*0.5f*(2.0f - ph_shared[place] - ph_shared[L])*(U_shared[place]-U_shared[L]) + 0.5f*(jat + jat_im)*nx;
    	 
    	 
        /*# ============================
        # top edge flux (i, j+1/2)
        # ============================*/     
        phx = phipjp - phimjp;
        phz = ph_shared[T]-ph_shared[place];
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nz = phz / sqrtf(phn2);}
                   else {nz = 0.0f;}    	
  
        float jat_jp = 0.5f*(1.0f+(1.0f-k)*U_shared[T])*(1.0f-ph_shared[T]*ph_shared[T])*dpsi_shared[T];      
        
        float UT = hi*Dl_tilde*0.5f*(2.0f - ph_shared[place] - ph_shared[T])*(U_shared[T]-U_shared[place]) + 0.5f*(jat + jat_jp)*nz;
    	 
    	 
        /*# ============================
        # bottom edge flux (i, j-1/2)
        # ============================*/  
        phx = phipjm - phimjm;
        phz = ph_shared[place]-ph_shared[B];
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nz = phz / sqrtf(phn2);}
                   else {nz = 0.0f;} 

        float jat_jm = 0.5f*(1.0f+(1.0f-k)*U_shared[B])*(1.0f-ph_shared[B]*ph_shared[B])*dpsi_shared[B];              
        float UB = hi*Dl_tilde*0.5f*(2.0f - ph_shared[place] - ph_shared[B])*(U_shared[place]-U_shared[B]) + 0.5f*(jat + jat_jm)*nz;
        
        float rhs_U = ( (UR-UL) + (UT-UB) ) * hi + cP.sqrt2 * jat;
        float tau_U = (1.0f+cP.k) - (1.0f-cP.k)*ph_shared[place];

        U_shared_new[place] = U_shared[place] + cP.dt * ( rhs_U / tau_U );
        U_new[C] = U_shared_new[place];
        
        
      }
   }
  __syncthreads();

  //  update psi and phi
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) ) {
      if (0<place<BLOCK_DIM_X -1) {
       // find the indices of the 8 neighbors for center
       int R=place+1;
       int L=place-1;
       int T=place+BLOCK_DIM_X;
       int B=place-BLOCK_DIM_X;
        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================

        // these ps's are defined on cell centers
        float psipjp=( ps_shared[place] + ps_shared[R] + ps_shared[T] + ps_shared[T+1] ) * 0.25f;
        float psipjm=( ps_shared[place] + ps_shared[R] + ps_shared[B] + ps_shared[B+1] ) * 0.25f;
        float psimjp=( ps_shared[place] + ps_shared[L] + ps_shared[T-1] + ps_shared[T] ) * 0.25f;
        float psimjm=( ps_shared[place] + ps_shared[L] + ps_shared[B-1] + ps_shared[B] ) * 0.25f;

        float phipjp=( ph_shared[place] + ph_shared[R] + ph_shared[T] + ph_shared[T+1] ) * 0.25f;
        float phipjm=( ph_shared[place] + ph_shared[R] + ph_shared[B] + ph_shared[B+1] ) * 0.25f;
        float phimjp=( ph_shared[place] + ph_shared[L] + ph_shared[T-1] + ph_shared[T] ) * 0.25f;
        float phimjm=( ph_shared[place] + ph_shared[L] + ph_shared[B-1] + ph_shared[B] ) * 0.25f;
        
        // if (C==132){
        //   printf("detailed check of neighbours 2\n");
        //   printf("R: %f ; L:%f ; T: %f ; B: %f \n", psipjp, psipjm, psimjp, psimjm);
        // }

        // ============================
        // right edge flux
        // ============================
        float psx = ps_shared[R]-ps_shared[place];
        float psz = psipjp - psipjm;
        float phx = ph_shared[R]-ph_shared[place];
        float phz = phipjp - phipjm;

        float A  = atheta( phx,phz);
        float Ap = aptheta(phx,phz);
        float JR = A * ( A*psx - Ap*psz );
        
        // ============================
        // left edge flux
        // ============================
        psx = ps_shared[place]-ps_shared[L];
        psz = psimjp - psimjm;
        phx = ph_shared[place]-ph_shared[L];
        phz = phimjp - phimjm; 

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        float JL = A * ( A*psx - Ap*psz );

        // ============================
        // top edge flux
        // ============================
        psx = psipjp - psimjp;
        psz = ps_shared[T]-ps_shared[place];
        phx = phipjp - phimjp;
        phz = ph_shared[T]-ph_shared[place];

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        float JT = A * ( A*psz + Ap*psx );

        // ============================
        // bottom edge flux
        // ============================
        psx = psipjm - psimjm;
        psz = ps_shared[place]-ps_shared[B];
        phx = phipjm - phimjm;
        phz = ph_shared[place]-ph_shared[B];

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        float JB = A * ( A*psz + Ap*psx );

         /*# =============================================================
        #
        # 2. EXTRA TERM: sqrt2 * atheta**2 * phi * |grad psi|^2
        #
        # =============================================================
        # d(phi)/dx  d(psi)/dx d(phi)/dz  d(psi)/dz at nodes (i,j)*/
        float phxn = ( ph_shared[R] - ph_shared[L] ) * 0.5f;
        float phzn = ( ph_shared[T] - ph_shared[B] ) * 0.5f;
        float psxn = ( ps_shared[R] - ps_shared[L] ) * 0.5f;
        float pszn = ( ps_shared[T] - ps_shared[B] ) * 0.5f;

        float A2 = atheta(phxn,phzn);
        A2 = A2*A2;
        float gradps2 = (psxn)*(psxn) + (pszn)*(pszn);
        float extra =  -cP.sqrt2 * A2 * ph_shared[place] * gradps2;

        /*# =============================================================
        #
        # 3. double well (transformed): sqrt2 * phi + nonlinear terms
        #
        # =============================================================*/

        float Up = (y[j]/cP.W0 - cP.R_tilde * (nt*cP.dt) )/cP.lT_tilde;

        float rhs_psi = ((JR-JL) + (JT-JB) + extra) * cP.hi*cP.hi + \
                   cP.sqrt2*ph_shared[place] - cP.lamd*(1.0f-ph_shared[place]*ph_shared[place])*cP.sqrt2*(U_shared_new[place] + Up);

        /*# =============================================================
        #
        # 4. dpsi/dt term
        #
        # =============================================================*/
        float tp = (1.0f-(1.0f-cP.k)*Up);
        float tau_psi;
        if (tp >= cP.k){tau_psi = tp*A2;}
               else {tau_psi = cP.k*A2;}
        
        dpsi_shared_new[place] = rhs_psi / tau_psi;
        ps_shared_new[place] = ps_shared[place] +  cP.dt * dpsi_shared_new[place];
        ph_shared_new[place] = tanhf(ps_shared_new[place]/cP.sqrt2);

        ps_new[C] = ps_shared_new[place];
        ph_new[C] = ph_shared_new[place];
        dpsi_new[C] = dpsi_shared_new[place];

         }
       }

}

// Host codes for PF computing
void setup(GlobalConstants params, int fnx, int fny, float* x, float* y, float* phi, float* psi,float* U){
  // we should have already pass all the data structure in by this time
  // move those data onto device
  printCudaInfo();
  float* x_device;// = NULL;
  float* y_device;// = NULL;
  // store two for swap behavior
  float* psi_old;// = NULL;
  float* psi_new;// = NULL;
  float* U_old;// = NULL;
  float* U_new;// = NULL;
  float* phi_old;// = NULL;
  float* phi_new;// = NULL;
  float* dpsi;// = NULL;
  float* dpsi_new;
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
  cudaMalloc((void **)&dpsi_new,    sizeof(float) * length);  


  // set initial params
  cudaMemcpy(x_device, x, sizeof(float) * fnx, cudaMemcpyHostToDevice);
  cudaMemcpy(y_device, y, sizeof(float) * fny, cudaMemcpyHostToDevice);
  cudaMemcpy(psi_old, psi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(phi_old, phi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(U_old, U, sizeof(float) * length, cudaMemcpyHostToDevice);

  // pass all the read-only params into global constant
  cudaMemcpyToSymbol(cP, &params, sizeof(GlobalConstants));

   int blocksize_1d = 128;
   int blocksize_2d = 128;  // seems reduce the block size makes it a little faster, but around 128 is okay.
   int num_block_2d = (fnx*fny+blocksize_2d-1)/blocksize_2d;
   int num_block_1d = (fnx+fny+blocksize_1d-1)/blocksize_1d;
   printf("nx: %d and ny: %d\n", fnx, fny);
   printf("block size %d, # blocks %d\n", blocksize_2d, num_block_2d); 
   // change the 2d block due to we donn't want to include halo region
   int num_block_sh = (fnx*(fny-2) + REAL_DIM - 1) / REAL_DIM;

   printf("blocks: %d and block_size: %d\n", num_block_sh, REAL_DIM);
   
   
   initialize<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, U_new, x_device, y_device, fnx, fny);
   set_BC<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_new, dpsi_new, fnx, fny);
   set_BC<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_old, dpsi, fnx, fny);
   rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, y_device, dpsi_new, fnx, fny, 0);
   cudaDeviceSynchronize();
   double startTime = CycleTimer::currentSeconds();

   for (int kt=0; kt<params.Mt/2; kt++){
  //  printf("time step %d\n",kt);
    set_BC<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_new, dpsi_new, fnx, fny);
 //   merge_PF<<< num_block_sh, BLOCK_DIM_X >>>(psi_new, phi_new, U_old, psi_old, phi_old, U_new, dpsi_new, dpsi, y_device, \
                    fnx, fny, 2*kt+1);

  //  rhs_U<<< num_block_2d, blocksize_2d >>>(U_old, U_new, phi_new, dpsi_new, fnx, fny);

    // //cudaDeviceSynchronize();
  //  rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_new, phi_new, U_new, psi_old, phi_old, y_device, dpsi, fnx, fny, 2*kt+1 );

    // //cudaDeviceSynchronize();
    set_BC<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_old, dpsi, fnx, fny);
 //   merge_PF<<< num_block_sh, BLOCK_DIM_X >>>(psi_old, phi_old, U_new, psi_new, phi_new, U_old, dpsi, dpsi_new, y_device, \
      fnx, fny, 2*kt+2);
  }

   cudaDeviceSynchronize();
   double endTime = CycleTimer::currentSeconds();
   printf("time for %d iterations: %f s\n", params.Mt, endTime-startTime);
   cudaMemcpy(psi, psi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(phi, phi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(U, U_old, length * sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(x_device); cudaFree(y_device);
  cudaFree(psi_old); cudaFree(psi_new);
  cudaFree(phi_old); cudaFree(phi_new);
  cudaFree(U_old); cudaFree(U_new);
  cudaFree(dpsi); cudaFree(dpsi_new); 


}





void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}

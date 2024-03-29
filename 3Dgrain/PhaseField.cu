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

  


__constant__ GlobalConstants cP;


// Device codes 

// boundary condition
// only use this function to access the boundary points, 
// other functions return at the boundary

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
set_minus1(float* u, int size){

     int index = blockIdx.x * blockDim.x + threadIdx.x;
     if(index<size) u[index] = -1.0f;

}


__global__ void
collect_PF(float* PFs, float* phi, int* alpha_m, int* argmax){

   int C = blockIdx.x * blockDim.x + threadIdx.x;
   int length = cP.length;
 //  int PF_id = index/(length);
 //  int C = index - PF_id*length;
   
  //if ( (C<length) && (PF_id<NUM_PF) ) {

    //if (C==513*717+716){printf("PF_id %d, values %f\n", PF_id, PFs[index]);}
   if (C<length){
   // for loop to find the argmax of the phase field
     for (int PF_id=0; PF_id<cP.NUM_PF; PF_id++){
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
ini_PF(float* PFs, float* phi, int* alpha_m){

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int length = cP.length;
   int PF_id = index/(length);
   int C = index - PF_id*length;

 // if (PF_id==1){
  if ( (C<length) && (PF_id<cP.NUM_PF) ) {
    if ( (phi[C]>LS) && (PF_id== (alpha_m[C]-1)) ){
  //  if ( (phi[C]>LS) && (PF_id== 0)){ //(alpha_m[C]-1)) ){
      PFs[index] = phi[C];
    }
 // }
  }
}



__global__ void
set_BC_3D(float* ph, int max_area){

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

  }

}


// anisotropy functions





// psi equation
__global__ void
rhs_psi(float* x, float* y, float* z, float* ph, float* ph_new, int nt, float t, \
       float* X, float* Y, float* Z, float* Tmac, float* u_3d, int Nx, int Ny, int Nz, int Nt, curandState* states, float* cost, float* sint){

  int C = blockIdx.x * blockDim.x + threadIdx.x; 
  int i, j, k, PF_id;
  int fnx = cP.fnx, fny = cP.fny, fnz = cP.fnz, NUM_PF = cP.NUM_PF;
  G2L_4D(C, i, j, k, PF_id, fnx, fny, fnz);
  int pf_C = C - PF_id*fnx*fny*fnz;
  // macros
  /*
   float Dt = Tmac[1]-Tmac[0];
   int kt = (int) ((t-Tmac[0])/Dt);
  // printf("%d ",kt);
   float delta_t = (t-Tmac[0])/Dt-kt;
   //printf("%f ",Dt);
   float Dx = X[1]-X[0]; // (X[Nx-1]-X[0]) / (Nx-1)
   float Dy = Y[1]-Y[0];
   float Dz = Z[1]-Z[0];
   */
  // if the points are at boundary, return
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) && (k>0) && (k<fnz-1) &&(PF_id<NUM_PF) ) {
       // find the indices of the 8 neighbors for center
      //if ( (ph[C]<1.0f) && (ph[C]>-1.0f) ){

       int R=C+1;
       int L=C-1;
       int T=C+fnx;
       int B=C-fnx;
       int U=C+fnx*fny;
       int D=C-fnx*fny;


       // first checkout the gradient 
       float phxn = ( ph[R] - ph[L] ) * 0.5f;
       float phyn = ( ph[T] - ph[B] ) * 0.5f;
       float phzn = ( ph[U] - ph[D] ) * 0.5f;


      // float gradph2 = phxn*phxn + phyn*phyn + phzn*phzn;
      // if (gradph2>1e-12){


       //float alpha = theta_arr[PF_id+1];
       float cosa, sina, cosb, sinb;
       if (ph[C]>LS){
       sina = sint[PF_id+1];
       cosa = cost[PF_id+1];
       sinb = sint[PF_id+1+NUM_PF];
       cosb = cost[PF_id+1+NUM_PF];
       }else{
       sina = 0.0f;
       cosa = 1.0f;
       sinb = 0.0f;
       cosb = 1.0f;
       }


  

        float A2 = kine_ani(phxn,phyn,phzn,cosa,sina,cosb,sinb);


        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================



        float diff =  ph[R] + ph[L] + ph[T] + ph[B] + ph[U] + ph[D] - 6*ph[C];
        


        /*# =============================================================
        #
        # 3. double well (transformed): sqrt2 * phi + nonlinear terms
        #
        # =============================================================*/
        //float Up = (y[j]/cP.W0 - cP.R_tilde * (nt*cP.dt) )/cP.lT_tilde;
      /*  
      int kx = (int) (( x[i] - X[0] )/Dx);
      float delta_x = ( x[i] - X[0] )/Dx - kx;
         //printf("%f ",delta_x);
      int ky = (int) (( y[j] - Y[0] )/Dy);
      float delta_y = ( y[j] - Y[0] )/Dy - ky;

      int kz = (int) (( z[j] - Z[0] )/Dz);
      float delta_z = ( z[j] - Z[0] )/Dz - kz;
      //printf("%d ",kx);
      if (kx==Nx-1) {kx = Nx-2; delta_x =1.0f;}
      if (ky==Ny-1) {ky = Ny-2; delta_y =1.0f;}
      if (kz==Nz-1) {kz = Nz-2; delta_z =1.0f;}      
      if (kt==Nt-1) {kt = Nt-2; delta_t =1.0f;}
      int offset    = L2G_4D(kx, ky, kz, kt, Nx, Ny, Nz);
      int offset_z  = L2G_4D(kx, ky, kz+1, kt, Nx, Ny, Nz);
      int offset_t  = L2G_4D(kx, ky, kz, kt+1, Nx, Ny, Nz);
      int offset_zt = L2G_4D(kx, ky, kz+1, kt+1, Nx, Ny, Nz);
      //if (offset_n>Nx*Ny*Nt-1-1-Nx) printf("%d, %d, %d, %d  ", i,j,kx,ky);
     // printf("%d ", Nx);
      float Tinterp= ( ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset ] + (1.0f-delta_x)*delta_y*u_3d[ offset+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset+1 ] +   delta_x*delta_y*u_3d[ offset+Nx+1 ] )*(1.0f-delta_z) + \
             ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset_z ] + (1.0f-delta_x)*delta_y*u_3d[ offset_z+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset_z+1 ] +   delta_x*delta_y*u_3d[ offset_z+Nx+1 ] )*delta_z )*(1.0f-delta_t) +\

                   + ( ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset_t ] + (1.0f-delta_x)*delta_y*u_3d[ offset_t+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset_t+1 ] +   delta_x*delta_y*u_3d[ offset_t+Nx+1 ] )*(1.0f-delta_z) + \
             ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset_zt ] + (1.0f-delta_x)*delta_y*u_3d[ offset_zt+Nx ] \
               +delta_x*(1.0f-delta_y)*u_3d[ offset_zt+1 ] +   delta_x*delta_y*u_3d[ offset_zt+Nx+1 ] )*delta_z )*delta_t;
       */
        float Tinterp = cP.G*(z[k] - cP.R*1e6 *t - 2);
        float Up = Tinterp/(cP.L_cp);  //(y[j]/cP.W0 - cP.R_tilde * (nt*cP.dt) )/cP.lT_tilde;
       // float Up = (Tinterp-cP.Ti)/(cP.c_infm/cP.k)/(1.0-cP.k);  //(y[j]/cP.W0 - cP.R_tilde * (nt*cP.dt) )/cP.lT_tilde;
        float repul=0.0f;
        for (int pf_id=0; pf_id<NUM_PF; pf_id++){
            
           if (pf_id!=PF_id) {
               int overlap_id = pf_C + pf_id*fnx*fny*fnz;
               repul += 0.25f*(ph[overlap_id]+1.0f)*(ph[overlap_id]+1.0f);
           }
        }
        float rhs_psi = diff * cP.hi*cP.hi + (1.0f-ph[C]*ph[C])*ph[C] \
              - cP.lamd*Up* ( (1.0f-ph[C]*ph[C])*(1.0f-ph[C]*ph[C]) - 0.5f*OMEGA*(ph[C]+1.0f)*repul);

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










void calc_qois(GlobalConstants params, QOI* q, int &cur_tip, int* alpha, int kt, float* z, int* loss_area, int move_count){

     // cur_tip here inludes the halo
     int fnx = params.fnx, fny = params.fny, fnz = params.fnz, num_grains = params.num_theta, all_time = params.nts+1;
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

}


__global__ void 
move_frame(float* ph_buff, float* z_buff, float* ph, float* z, int* alpha, int* alpha_full, int* loss_area, int move_count){

    int C = blockIdx.x * blockDim.x + threadIdx.x;
    int i, j, k, PF_id;
    int fnx = cP.fnx, fny = cP.fny, fnz = cP.fnz, NUM_PF = cP.NUM_PF;
    G2L_4D(C, i, j, k, PF_id, fnx, fny, fnz);

    if ( (i==0) && (j==0) && (k>0) && (k<fnz-2) && (PF_id==0) ) {
        z_buff[k] = z[k+1];}

    if ( (i==0) && (j==0) &&  (k==fnz-2) && (PF_id==0) ) {        
        z_buff[k] = 2*z[fnz-2] - z[fnz-3];}

    if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) && (k>0) && (k<fnz-2) && (PF_id<NUM_PF) ) {     
        ph_buff[C] = ph[C+fnx*fny];}

    if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) && (k==fnz-2) && (PF_id<NUM_PF) ) {

        ph_buff[C] = 2*ph[C] - ph[C-fnx*fny];}

    // add last layer of alpha to alpha_full[move_count]
    if ( (i<fnx) && (j<fny) && (k==1) && (PF_id==0) ) {

        alpha_full[move_count*fnx*fny+C] = alpha[C];
        atomicAdd(loss_area+alpha[C]-1,1);
        //printf("%d ", alpha[C]);
    }


}

__global__ void
copy_frame(float* ph_buff, float* z_buff, float* ph, float* z){

  int C = blockIdx.x * blockDim.x + threadIdx.x; 
  int i, j, k, PF_id;
  int fnx = cP.fnx, fny = cP.fny, fnz = cP.fnz, NUM_PF = cP.NUM_PF;
  G2L_4D(C, i, j, k, PF_id, fnx, fny, fnz);

  if ( (i == 0) && (j==0) && (k>0) && (k < fnz-1) && (PF_id==0) ){
        z[k] = z_buff[k];}

  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) && (k>0) && (k<fnz-1) && (PF_id<NUM_PF) ) { 
        ph[C] = ph_buff[C];}


}


__global__ void
ave_x(float* phi, float* meanx, int fnx, int fny, int fnz, int NUM_PF){


  int C = blockIdx.x * blockDim.x + threadIdx.x; 
  int i, j, k, pf_id;
  G2L_4D(C, i, j, k, pf_id, fnx, fny, fnz);
   if (C<fnx*fny*fnz*NUM_PF){
      atomicAdd(meanx+k,phi[C]);

   } 

}



void tip_mvf(int *cur_tip, float* phi, float* meanx, float* meanx_host, int fnx, int fny, int fnz, int NUM_PF){

     int length = fnx*fny*fnz;
     int blocksize_2d = 128; 
     int num_block_PF = (length*NUM_PF+blocksize_2d-1)/blocksize_2d;

     ave_x<<<num_block_PF, blocksize_2d>>>(phi, meanx, fnx, fny, fnz, NUM_PF);

     cudaMemcpy(meanx_host, meanx, fnz * sizeof(float),cudaMemcpyDeviceToHost);
     while( (meanx_host[*cur_tip]/(NUM_PF*fnx*fny)>LS) && (*cur_tip<fnz-1) ) {
      *cur_tip+=1;
      //printf("average ph %f along location %d\n", meanx_host[*cur_tip]/(NUM_PF*fnx*fny), *cur_tip);
      }
    // for (int ww=0; ww<fny; ww++){ printf("avex %f \n",meanx_host[ww]/fnx);}
//      printf("currrent tip %d \n", *cur_tip);   
     cudaMemset(meanx,0,fnz * sizeof(float));

}


PhaseField::~PhaseField() {
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


void PhaseField::cudaSetup(params_MPI pM) {

    int num_gpus_per_node = 4;
    int device_id_innode = pM.rank % num_gpus_per_node;
    //gpu_name = cuda.select_device( )
    cudaSetDevice(device_id_innode); 
    printCudaInfo(pM.rank,device_id_innode);
    params.NUM_PF = params.num_theta;
    NUM_PF = params.NUM_PF;
    
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
    cudaMalloc((void **)&argmax,    sizeof(int) * length);
    cudaMemset(argmax,0,sizeof(int) * length);

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





void PhaseField::evolve(){
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
   ini_PF<<< num_block_PF, blocksize_2d >>>(PFs_old, phi_old, alpha_m);
   ini_PF<<< num_block_PF, blocksize_2d >>>(PFs_new, phi_old, alpha_m);
   set_minus1<<< num_block_2d, blocksize_2d >>>(phi_old,length);

     //cudaDeviceSynchronize();
   set_BC_3D<<<num_block_PF1d, blocksize_1d>>>(PFs_new, max_area);
   set_BC_3D<<<num_block_PF1d, blocksize_1d>>>(PFs_old, max_area);

   rhs_psi<<< num_block_PF, blocksize_2d >>>(x_device, y_device, z_device, PFs_old, PFs_new, 0, 0, \
    Mgpu.X_mac, Mgpu.Y_mac,  Mgpu.Z_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nz, mac.Nt, dStates, Mgpu.cost, Mgpu.sint);

   calc_qois(params, q, cur_tip, alpha, 0, z, loss_area, move_count);
   cudaDeviceSynchronize();
   double startTime = CycleTimer::currentSeconds();
   for (int kt=0; kt<params.Mt/2; kt++){

     set_BC_3D<<<num_block_PF1d, blocksize_1d>>>(PFs_new, max_area);

     t_cur_step = (2*kt+1)*params.dt*params.tau0;
     rhs_psi<<< num_block_PF, blocksize_2d >>>(x_device, y_device, z_device, PFs_new, PFs_old, 2*kt+1,t_cur_step, \
        Mgpu.X_mac, Mgpu.Y_mac, Mgpu.Z_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nz, mac.Nt, dStates, Mgpu.cost, Mgpu.sint);
 
     set_BC_3D<<<num_block_PF1d, blocksize_1d>>>(PFs_old, max_area);

    if ( (2*kt+2)%kts==0) {
             //tip_mvf(&cur_tip,phi_new, meanx, meanx_host, fnx,fny);
             cudaMemset(alpha_m, 0, sizeof(int) * length);
             collect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, argmax);
             cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost); 
             cudaMemcpy(loss_area, d_loss_area, params.num_theta * sizeof(int),cudaMemcpyDeviceToHost);
             cudaMemcpy(z, z_device, fnz * sizeof(int),cudaMemcpyDeviceToHost); 
             //QoIs based on alpha field
             cur_tip=0;
             calc_qois(params, q, cur_tip, alpha, (2*kt+2)/kts, z, loss_area, move_count);
          }
     //if ( (2*kt+2)%params.ha_wd==0 )commu_BC(comm, SR_buffs, pM, 2*kt+1, params.ha_wd, fnx, fny, psi_old, phi_old, U_new, dpsi, alpha_m);
     //cudaDeviceSynchronize();
     if ( (2*kt+2)%TIPP==0) {
             tip_mvf(&tip_front, PFs_old, meanx, meanx_host, fnx,fny,fnz,NUM_PF);
           //  lowsl = 1;
             collect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, argmax);
             cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
            // sampleh(&lowsl, alpha, fnx,fny);
             //printf("lowsl %d\n", lowsl);
             while (tip_front >=tip_thres){
                collect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, argmax);
                move_frame<<< num_block_PF, blocksize_2d >>>(PFs_new, z_device2, PFs_old, z_device, alpha_m, d_alpha_full, d_loss_area, move_count);
                copy_frame<<< num_block_PF, blocksize_2d >>>(PFs_new, z_device2, PFs_old, z_device);
                move_count +=1;
                tip_front-=1;

             }

          set_BC_3D<<<num_block_PF1d, blocksize_1d>>>(PFs_old, max_area);
          }

     t_cur_step = (2*kt+2)*params.dt*params.tau0;
     rhs_psi<<< num_block_PF, blocksize_2d >>>(x_device, y_device, z_device, PFs_old, PFs_new,  2*kt+2,t_cur_step, \
        Mgpu.X_mac, Mgpu.Y_mac, Mgpu.Z_mac, Mgpu.t_mac, Mgpu.T_3D, mac.Nx, mac.Ny, mac.Nz, mac.Nt, dStates, Mgpu.cost, Mgpu.sint);


   }


   cudaDeviceSynchronize();
   double endTime = CycleTimer::currentSeconds();
   printf("time for %d iterations: %f s\n", params.Mt, endTime-startTime);

   
   cudaMemset(alpha_m, 0, sizeof(int) * length);
   collect_PF<<< num_block_2d, blocksize_2d >>>(PFs_old, phi_old, alpha_m, argmax); 
   cudaMemcpy(phi, phi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(alpha, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
   cudaMemcpy(alpha_i_full, d_alpha_full, fnx*fny*fnz_f * sizeof(int),cudaMemcpyDeviceToHost);
   cudaMemcpy(alpha_i_full+move_count*fnx*fny, alpha_m, length * sizeof(int),cudaMemcpyDeviceToHost);
 //  calc_frac(alpha, fnx, fny, fnz, params.nts, params.num_theta, tip_y, frac, z, aseq, ntip, left_coor);

}









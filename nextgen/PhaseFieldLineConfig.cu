#include "APTPhaseField.h"


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
APTmove_frame(float* ph_buff, int* arg_buff, float* z_buff, float* ph, int* arg, float* z, int* alpha, int* alpha_full, int* loss_area, int move_count)
{

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
APTcopy_frame(float* ph_buff, int* arg_buff, float* z_buff, float* ph, int* arg, float* z)
{

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



void APTPhaseField::getLineQoIs(MovingDomain* movingDomainManager)
{
    int num_block_2d = (length+blocksize_2d-1)/blocksize_2d;
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
    int max_area = max(fnz*fny,max(fny*fnx,fnx*fnz));
    int num_block_PF1d =  ( max_area*NUM_PF +blocksize_1d-1)/blocksize_1d;
    int num_block_2d = (length+blocksize_2d-1)/blocksize_2d;
    int num_block_PF = (length*NUM_PF+blocksize_2d-1)/blocksize_2d;

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

void MovingDomain::allocateMovingDomain(int numGrains, int MovingDirectoinSize)
{
    tip_thres = (int) ((1-BLANK)*MovingDirectoinSize);
    printf("max tip can go: %d\n", tip_thres); 

    meanx_host = new float[MovingDirectoinSize];
    loss_area_host = new int[numGrains];
    memset(loss_area_host, 0, sizeof(int) * numGrains);

    cudaMalloc((void **)&meanx_device, sizeof(float) * MovingDirectoinSize);
    cudaMemset(meanx_device,0, sizeof(float) * MovingDirectoinSize);

    cudaMalloc((void **)&loss_area_device, sizeof(int) * numGrains); 
    cudaMemset(loss_area_device, 0, sizeof(int) * numGrains);
}
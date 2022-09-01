#include "params.h"

// constructor

PhaseField::PhaseField() {

    x = NULL;
    phi = NULL;
    x_device = NULL;
    phi_new = NULL;
}


void PhaseField::cpuSetup(params_MPI pM){


    pM.nx_loc = params.nx/pM.nprocx;
    pM.ny_loc = params.ny/pM.nprocy;
    pM.nz_loc = params.nz;
    pM.nz_full_loc = params.nz_full;
    
    float dxd = params.dx*params.W0;
    float len_blockx = pM.nx_loc*dxd; //params.lxd/pM.nprocx;
    float len_blocky = pM.ny_loc*dxd; //params.lyd/pM.nprocy;
    
    float xmin_loc = params.xmin+ pM.px*len_blockx; 
    float ymin_loc = params.ymin+ pM.py*len_blocky; 
    float zmin_loc = params.zmin;

    if (pM.px==0){pM.nx_loc+=1;}
    else{xmin_loc+=dxd;} //params.lxd/params.nx;}
    
    if (pM.py==0){pM.ny_loc+=1;}
    else{ymin_loc+=dxd;}//params.lyd/params.ny;}

    pM.nz_loc+=1;
    pM.nz_full_loc+=1;

    params.fnx = pM.nx_loc+2*params.ha_wd;
    params.fny = pM.ny_loc+2*params.ha_wd;
    params.fnz = pM.nz_loc+2*params.ha_wd;
    params.fnz_f = pM.nz_full_loc+2*params.ha_wd;
    fnx = params.fnx, fny = params.fny, fnz = params.fnz;
    params.length=fnx*fny*fnz;
    params.full_length = fnx*fny*fnz_f;
    length = params.length,full_length = params.full_length, NUM_PF = params.NUM_PF;

    x = new float[fnx];
    y = new float[fny];
    z = new float[fnz];
    z_full = new float[fnz_f];

    for(int i=0; i<fnx; i++){
        x[i]=(i-params.ha_wd)*dxd + xmin_loc; 
    }

    for(int i=0; i<fny; i++){
        y[i]=(i-params.ha_wd)*dxd + ymin_loc;
    }

    for(int i=0; i<fnz; i++){
        z[i]=(i-params.ha_wd)*dxd + zmin_loc;
    }

    for(int i=0; i<fnz_f; i++){
        z_full[i]=(i-params.ha_wd)*dxd + zmin_loc;
    }


    psi = new float[length];
    phi = new float[length];
  //  Uc = new float[length];
    alpha_i = new int[length];
    alpha_i_full = new int[full_length];

  //  std::cout<<"x= ";
  //  for(int i=0; i<fnx; i++){
  //      std::cout<<x[i]<<" ";
  //  }
    cout<< "rank "<< pM.rank<< " xmin " << pf_solver->x[0] << " xmax "<<pf_solver->x[fnx-1]<<endl;
    cout<< "rank "<< pM.rank<< " ymin " << pf_solver->y[0] << " ymax "<<pf_solver->y[fny-1]<<endl;
    cout<< "rank "<< pM.rank<< " zmin " << pf_solver->z[0] << " zmax "<<pf_solver->z[fnz-1]<<endl;
    cout<<"x length of psi, phi, U="<<fnx<<endl;
    cout<<"y length of psi, phi, U="<<fny<<endl;
    cout<<"z length of psi, phi, U="<<fnz<<endl;   
    cout<<"length of psi, phi, U="<<length<<endl;
 

}


void PhaseField::initField(Mac_input mac){

    for (int i=0; i<2*NUM_PF+1; i++){
        //mac.theta_arr[i+1] = 1.0f*(rand()%10)/(10-1)*(-M_PI/2.0);
       // mac.theta_arr[i+1] = 1.0f*rand()/RAND_MAX*(-M_PI/2.0);
       // mac.theta_arr[i+1] = (i)*grain_gap- M_PI/2.0;
        mac.sint[i] = sinf(mac.theta_arr[i]);
        mac.cost[i] = cosf(mac.theta_arr[i]);
    }  
   
    for (int i=0; i<params.num_theta; i++){
       aseq[i] = i+1;} //rand()%NUM_PF +1;
     
    float Dx = mac.X_mac[mac.Nx-1] - mac.X_mac[mac.Nx-2];
    float Dy = mac.Y_mac[mac.Ny-1] - mac.Y_mac[mac.Ny-2];
    float Dz = mac.Z_mac[mac.Nz-1] - mac.Z_mac[mac.Nz-2];    
    for(int id=0; id<length; id++){
      int k = id/(fnx*fny);
      int k_r = id - k*fnx*fny;
      int j = k_r/fnx;
      int i = k_r%fnx; 
      
      if ( (i>params.ha_wd-1) && (i<fnx-params.ha_wd) && (j>params.ha_wd-1) && (j<fny-params.ha_wd) && (k>params.ha_wd-1) && (k<fnz-params.ha_wd)){
      int kx = (int) (( x[i] - mac.X_mac[0] )/Dx);
      float delta_x = ( x[i] - mac.X_mac[0] )/Dx - kx;
         //printf("%f ",delta_x);
      int ky = (int) (( y[j] - mac.Y_mac[0] )/Dy);
      float delta_y = ( y[j] - mac.Y_mac[0] )/Dy - ky;

      int kz = (int) (( z[k] - mac.Z_mac[0] )/Dz);
      float delta_z = ( z[k] - mac.Z_mac[0] )/Dz - kz;

      if (kx==mac.Nx-1) {kx = mac.Nx-2; delta_x =1.0f;}
      if (ky==mac.Ny-1) {ky = mac.Ny-2; delta_y =1.0f;}
      if (kz==mac.Nz-1) {kz = mac.Nz-2; delta_z =1.0f;}

      int offset =  kx + ky*mac.Nx + kz*mac.Nx*mac.Ny;
      int offset_n =  kx + ky*mac.Nx + (kz+1)*mac.Nx*mac.Ny;
      //if (offset>mac.Nx*mac.Ny-1-1-mac.Nx) printf("%d, %d, %d, %d  ", i,j,kx,ky);
      psi[id] = ( (1.0f-delta_x)*(1.0f-delta_y)*mac.psi_mac[ offset ] + (1.0f-delta_x)*delta_y*mac.psi_mac[ offset+mac.Nx ] \
               +delta_x*(1.0f-delta_y)*mac.psi_mac[ offset+1 ] +   delta_x*delta_y*mac.psi_mac[ offset+mac.Nx+1 ] )*(1.0f-delta_z) + \
             ( (1.0f-delta_x)*(1.0f-delta_y)*mac.psi_mac[ offset_n ] + (1.0f-delta_x)*delta_y*mac.psi_mac[ offset_n+mac.Nx ] \
               +delta_x*(1.0f-delta_y)*mac.psi_mac[ offset_n+1 ] +   delta_x*delta_y*mac.psi_mac[ offset_n+mac.Nx+1 ] )*delta_z;

  
     //   psi[id]=0.0;
      psi[id] = psi[id]/params.W0;
      phi[id]=tanhf(psi[id]/params.sqrt2);
     //   Uc[id]=0.0;
      if (phi[id]>LS){

        alpha_i[id] = alpha_cross[(j-1)*pM.nx_loc+(i-1)];
       if (alpha_i[id]<1 || alpha_i[id]>NUM_PF) cout<<(j-1)*pM.nx_loc+(i-1)<<alpha_i[id]<<endl;
       }

      else {alpha_i[id]=0;}
      }

    else{
       psi[id]=0.0f;
       phi[id]=0.0f;
     //  Uc[id]=0.0f;
       alpha_i[id]=0;
 
    }
    }



}








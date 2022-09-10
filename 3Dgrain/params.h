#ifndef PARAMS_H
#define PARAMS_H 
#include <string>
struct GlobalConstants {
  int nx, ny, nz, nz_full, fnx, fny, fnz, fnz_f, length, full_length, ha_wd;
  int Mt, nts, num_theta, NUM_PF, num_nodes; 
  float lx, lxd, lyd, lzd, tmax;
  float xmin, ymin, zmin; // MPI-related
  float asp_ratio_yx, asp_ratio_zx, moving_ratio;
  float dx, dt, hi, cfl, dt_sqrt;
  int ictype;
  float G, R, delta, kin_delta;
  float k, c_infty, m_slope, c_infm, Dl, GT;
  float Dh, d0, W0, L_cp;
  float lamd, tau0, beta0, mu_k, lT;
  float R_tilde, Dl_tilde, lT_tilde, beta0_tilde;
  float alpha0, U0, eps, eta;  

  // parameters that are not in the input file
  float cosa, sina, sqrt2, a_s, epsilon, a_12;
  int noi_period, seed_val; // noise
  // nucleation parameters
  float Tmelt, Ti, Tliq, Tsol;
  float undcool_mean, undcool_std, nuc_Nmax, nuc_rad, pts_cell; 

};

struct params_MPI{

    int rank;
    int px, py, pz;
    int nproc, nprocx, nprocy, nprocz;
    int nx_loc, ny_loc, nz_loc, nz_full_loc;
};

struct Mac_input{
  std::string folder;
  int Nx,  Ny, Nz, Nt;
  float* X_mac; 
  float* Y_mac; 
  float* Z_mac;
  float* t_mac;
  int* alpha_mac;
  float* psi_mac;
  float* U_mac;
  float* T_3D;
  float* theta_arr;
  float* cost;
  float* sint;
};



#endif 











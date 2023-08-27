//////////////////////
// params.h
// COPYRIGHT Yigong Qin, ALL RIGHTS RESERVED
/////////////////////

#pragma once

#include <string>

struct GlobalConstants {

  // processing parameters
  float G, R, delta, kin_delta;

  // physical parameters
  float k, c_infty, m_slope, c_infm, Dl, GT;
  float Dh, d0, W0, L_cp;
  float lamd, tau0, beta0, mu_k, lT;
  float R_tilde, Dl_tilde, lT_tilde, beta0_tilde;
  float alpha0, U0, eps, eta;  

  // geometry parameters
  float lx, lxd, lyd, lzd, tmax;
  float xmin, ymin, zmin; 
  float asp_ratio_yx, asp_ratio_zx, moving_ratio;

  // grid parameters
  int nx, ny, nz, nz_full, fnx, fny, fnz, fnz_f, length, full_length, haloWidth;
  int Mt, nts, num_theta, NUM_PF, num_nodes; 
  float dx, dt, hi, cfl, dt_sqrt;

  // initial condition parameters
  int ictype;

  // anisotropy parameters
  float cosa, sina, sqrt2, a_s, epsilon, a_12;

  // noise parameters
  int noi_period, seed_val; 

  // nucleation parameters
  float Tmelt, Ti, Tliq, Tsol;
  float undcool_mean, undcool_std, nuc_Nmax, nuc_rad, pts_cell; 

  // sampling parameters
  float z0, top;
  int num_samples;
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










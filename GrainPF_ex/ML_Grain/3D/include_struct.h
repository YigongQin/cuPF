struct GlobalConstants {
  int nx;
  int ny;
  int nz;
  int Mt;
  int nts; 
  int ictype;
  float G;
  float R;
  float delta;
  float k;
  float c_infm;
  float Dl;
  float Dh;
  float d0;
  float W0;
  float lT;
  float lamd; 
  float tau0;
  float c_infty; 
  float R_tilde;
  float Dl_tilde; 
  float lT_tilde; 
  float beta0_tilde;
  float eps; 
  float alpha0; 
  float dx; 
  float dt; 
  float asp_ratio_yx;
  float asp_ratio_zx; 
  float lxd;
  float lx; 
  float lyd; 
  float lzd;
  float eta; 
  float U0;
  float cfl; 
  float kin_delta;
  float beta0;
  float kin_coeff;
  float GT;
  float m_slope;
  float mu_k;
  float L_cp;
  int num_theta;
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
  float zmin;
 
  // nucleation parameters
  float Tmelt;
  float Tliq;
  float Tsol;
  float undcool_mean;
  float undcool_std;
  float nuc_Nmax;
  float nuc_rad;
  int pts_cell; 

  float moving_ratio;
  int nz_full;
  float tmax;
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
    int nz_loc;

    // moving-domain
    int nz_full_loc;
};

struct Mac_input{
  int Nx;
  int Ny;
  int Nz;
  int Nt;
  float* X_mac; 
  float* Y_mac; 
  float* Z_mac;
  float* t_mac;
  float* alpha_mac;
  float* psi_mac;
  float* U_mac;
  float* T_3D;
  float* theta_arr;
  float* cost;
  float* sint;
};















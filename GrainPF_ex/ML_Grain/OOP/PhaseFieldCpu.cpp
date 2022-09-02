#include "params.h"
#include "PhaseField.h"
#include "QOI.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <hdf5.h>
#include <random>
using namespace std;
#define LS -0.995
// constructor

PhaseField::PhaseField() {

    x = nullptr;
    phi = nullptr;
    x_device = nullptr;
    phi_new = nullptr;
    q = new QOI();
}

void getParam(std::string lineText, std::string key, float& param){
    std::stringstream iss(lineText);
    std::string word;
    while (iss >> word){
        //std::cout << word << std::endl;
        std::string myKey=key;
        if(word!=myKey) continue;
        iss>>word;
        std::string equalSign="=";
        if(word!=equalSign) continue;
        iss>>word;
        param=std::stof(word);
        
    }
}


void read_input(std::string input, float* target){
    std::string line;
    int num_lines = 0;
    std::ifstream graph(input);

    while (std::getline(graph, line))
        {
           std::stringstream ss(line);
           ss >> target[num_lines];
           num_lines+=1;}

}

void read_input(std::string input, int* target){
    std::string line;
    int num_lines = 0;
    std::ifstream graph(input);

    while (std::getline(graph, line))
        {
           std::stringstream ss(line);
           ss >> target[num_lines];
           num_lines+=1;}

}


void PhaseField::parseInputParams(Mac_input mac, char* fileName){

    float nts;
    float ictype;
    float ha_wd;
    float temp_Nx, temp_Ny, temp_Nz, temp_Nt;
    float seed_val, nprd;
    ifstream parseFile(fileName);
    string lineText;
    while (parseFile.good()){
        std::getline (parseFile, lineText);
        // Output the text from the file
        //getParam(lineText, "G", params.G);
        //getParam(lineText, "R", params.R); 
        getParam(lineText, "delta", params.delta); 
        getParam(lineText, "k", params.k); 
        //getParam(lineText, "c_infm", params.c_infm); 
        getParam(lineText, "Dh", params.Dh); 
        //getParam(lineText, "d0", params.d0); 
        getParam(lineText, "W0", params.W0);  
        getParam(lineText, "c_infty", params.c_infty);
        getParam(lineText, "m_slope", params.m_slope);
        //getParam(lineText, "beta0", params.beta0);
        getParam(lineText, "GT", params.GT);
        getParam(lineText, "L_cp", params.L_cp);
        getParam(lineText, "mu_k", params.mu_k);
        getParam(lineText, "eps", params.eps);
        getParam(lineText, "alpha0", params.alpha0);
        getParam(lineText, "dx", params.dx);
        getParam(lineText, "asp_ratio_yx", params.asp_ratio_yx);
        getParam(lineText, "asp_ratio_zx", params.asp_ratio_zx);
      //  getParam(lineText, "nx", nx);
      //  params.nx = (int)nx;
      //  getParam(lineText, "Mt", Mt);
      //  params.Mt = (int)Mt;
        getParam(lineText, "eta", params.eta);
        getParam(lineText, "U0", params.U0);
        getParam(lineText, "nts", nts);
        params.nts = (int)nts;
        getParam(lineText, "ictype", ictype);
        params.ictype = (int)ictype;
       getParam(lineText, "seed_val", seed_val);
        params.seed_val = (int)seed_val;
        getParam(lineText, "noi_period", nprd);
        params.noi_period = (int)nprd;
        getParam(lineText, "kin_delta", params.kin_delta);
      //  params.kin_delta = 0.05 + atoi(argv[3])/10.0*0.25;
        getParam(lineText, "beta0", params.beta0);
        // new multiple
        //getParam(lineText, "Ti", params.Ti);
        getParam(lineText, "ha_wd", ha_wd);
        params.ha_wd = (int)ha_wd;
        getParam(lineText, "xmin", params.xmin);
        getParam(lineText, "ymin", params.ymin);
        getParam(lineText, "zmin", params.zmin);
      //  getParam(lineText, "num_theta", num_thetaf);
      //  params.num_theta = (int) num_thetaf;
        getParam(lineText, "Nx", temp_Nx);
        getParam(lineText, "Ny", temp_Ny);
        getParam(lineText, "Nz", temp_Nz);
        getParam(lineText, "Nt", temp_Nt);
        getParam(lineText, "cfl", params.cfl); 

        getParam(lineText, "Tmelt", params.Tmelt);
        getParam(lineText, "undcool_mean", params.undcool_mean);
        getParam(lineText, "undcool_std", params.undcool_std);
        getParam(lineText, "nuc_Nmax", params.nuc_Nmax);
        getParam(lineText, "nuc_rad", params.nuc_rad);

        getParam(lineText, "moving_ratio", params.moving_ratio);
    }
    
    float dxd = params.dx*params.W0;
    // Close the file
    parseFile.close();


    float G0;
    float Rmax;
    float num_thetaf;
   
    
    mac.Nx = (int) temp_Nx;
    mac.Ny = (int) temp_Ny;
    mac.Nz = (int) temp_Nz;
    mac.Nt = (int) temp_Nt;
    mac.X_mac = new float[mac.Nx];
    mac.Y_mac = new float[mac.Ny];
    mac.Z_mac = new float[mac.Nz];
    mac.t_mac = new float[mac.Nt];
    
    mac.psi_mac = new float [mac.Nx*mac.Ny*mac.Nz];
    mac.U_mac = new float [mac.Nx*mac.Ny*mac.Nz];
    mac.T_3D = new float[mac.Nx*mac.Ny*mac.Nz*mac.Nt];
    

    read_input(params.mac_folder+"/x.txt", mac.X_mac);
    read_input(params.mac_folder+"/y.txt", mac.Y_mac);
    read_input(params.mac_folder+"/z.txt", mac.Z_mac);
    read_input(params.mac_folder+"/t.txt", mac.t_mac);
   // read_input(params.mac_folder+"/alpha.txt",mac.alpha_mac);
    read_input(params.mac_folder+"/psi.txt",mac.psi_mac);
    read_input(params.mac_folder+"/U.txt",mac.U_mac);
    read_input(params.mac_folder+"/G.txt", &G0);
    read_input(params.mac_folder+"/Rmax.txt", &Rmax);
    read_input(params.mac_folder+"/NG.txt", &num_thetaf);
    params.num_theta = (int) num_thetaf;
    params.NUM_PF = params.num_theta;
    int NUM_PF = params.NUM_PF;
    mac.theta_arr = new float[2*NUM_PF+1];
    mac.cost = new float[2*NUM_PF+1];
    mac.sint = new float[2*NUM_PF+1];
    mac.theta_arr[0] = 0.0f;
    read_input(params.mac_folder+"/theta.txt", mac.theta_arr);





    hid_t  h5in_file,  datasetT, dataspaceT, memspace;
    hsize_t dimT[1];
    herr_t  status;
    dimT[0] = mac.Nx*mac.Ny*mac.Nz*mac.Nt; 
    h5in_file = H5Fopen( (params.mac_folder+"/Temp.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    datasetT = H5Dopen2(h5in_file, "Temp", H5P_DEFAULT);
    dataspaceT = H5Dget_space(datasetT);
    memspace = H5Screate_simple(1,dimT,NULL);
    status = H5Dread(datasetT, H5T_NATIVE_FLOAT, memspace, dataspaceT,
                     H5P_DEFAULT, mac.T_3D);
    printf("mac.T %f\n",mac.T_3D[mac.Nx*mac.Ny*mac.Nz*mac.Nt-1]); 
    H5Dclose(datasetT);
    H5Sclose(dataspaceT);
    H5Sclose(memspace);
    H5Fclose(h5in_file);



    params.c_infm = params.c_infty*params.m_slope;
    params.Tliq = params.Tmelt - params.c_infm;
    params.Tsol = params.Tmelt - params.c_infm/params.k;
    params.Ti = params.Tsol;   
    params.d0 = params.GT/params.L_cp;
    params.beta0 = 1.0/(params.mu_k*params.L_cp);
    //params.lT = params.c_infm*( 1.0/params.k-1 )/params.G;//       # thermal length           um
    params.lamd = 0.8839*params.W0/params.d0;//     # coupling constant
    //params.tau0 = 0.6267*params.lamd*params.W0*params.W0/params.Dl; //    # time scale  
    params.tau0 = params.beta0*params.lamd*params.W0/0.8839;
    //params.kin_coeff = tauk/params.tau0;
    //params.R_tilde = params.R*params.tau0/params.W0;
    //params.Dl_tilde = params.Dl*params.tau0/pow(params.W0,2);
    //params.lT_tilde = params.lT/params.W0;
    params.beta0_tilde = params.beta0*params.W0/params.tau0;
    params.dt = params.cfl*params.dx*params.beta0_tilde;
//    params.ny = (int) (params.asp_ratio*params.nx);
    params.lxd = mac.X_mac[mac.Nx-2]; //-params.xmin; //this has assumption of [,0] params.dx*params.W0*params.nx; # horizontal length in micron
//    params.lyd = params.asp_ratio*params.lxd;
    params.hi = 1.0/params.dx;
    params.cosa = cos(params.alpha0/180*M_PI);
    params.sina = sin(params.alpha0/180*M_PI);
    params.sqrt2 = sqrt(2.0);
    params.a_s = 1 - 3.0*params.delta;
    params.epsilon = 4.0*params.delta/params.a_s;
    params.a_12 = 4.0*params.a_s*params.epsilon;
    params.dt_sqrt = sqrt(params.dt);

    params.nx = (int) (params.lxd/params.dx/params.W0);//global cells 
    params.ny = (int) (params.asp_ratio_yx*params.nx);
    params.nz = (int) (params.moving_ratio*params.nx);
    params.nz_full = (int) (params.asp_ratio_zx*params.nx);
    params.lxd = params.nx*dxd;
    params.lyd = params.ny*dxd;
    params.lzd = params.nz*dxd;
    params.Mt = (int) (mac.t_mac[mac.Nt-1]/params.tau0/params.dt);
    params.Mt = (params.Mt/2)*2; 
    int kts = params.Mt/params.nts;
    kts = (kts/2)*2;
    params.Mt = kts*params.nts;
    params.pts_cell = (int) (params.nuc_rad/dxd);

    params.G = G0;
    params.R = Rmax;
    params.tmax = params.tau0*params.dt*params.Mt;



    if (pM.rank==0){ 
        // print params and mac information

    std::cout<<"dx = "<<params.lxd/params.nx/params.W0<<std::endl;
    std::cout<<"dy = "<<params.lyd/params.ny/params.W0<<std::endl;   
    std::cout<<"dz = "<<params.lyd/params.ny/params.W0<<std::endl;  


    std::cout<<"noise coeff = "<<params.dt_sqrt*params.hi*params.eta<<std::endl;


    std::cout<<"mac Nx = "<<mac.Nx<<std::endl;
    std::cout<<"mac Ny = "<<mac.Ny<<std::endl;
    std::cout<<"mac Nz = "<<mac.Nz<<std::endl;
    std::cout<<"mac Nt = "<<mac.Nt<<std::endl;


    }
}

void PhaseField::cpuSetup(params_MPI &pM){


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
    fnx = params.fnx, fny = params.fny, fnz = params.fnz, fnz_f = params.fnz_f;
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
    alpha = new int[length];
    alpha_i_full = new int[full_length];

  //  std::cout<<"x= ";
  //  for(int i=0; i<fnx; i++){
  //      std::cout<<x[i]<<" ";
  //  }
    cout<< "rank "<< pM.rank<< " xmin " << x[0] << " xmax "<<x[fnx-1]<<endl;
    cout<< "rank "<< pM.rank<< " ymin " << y[0] << " ymax "<<y[fny-1]<<endl;
    cout<< "rank "<< pM.rank<< " zmin " << z[0] << " zmax "<<z[fnz-1]<<endl;
    cout<<"x length of psi, phi, U="<<fnx<<endl;
    cout<<"y length of psi, phi, U="<<fny<<endl;
    cout<<"z length of psi, phi, U="<<fnz<<endl;   
    cout<<"length of psi, phi, U="<<length<<endl;
 
    q->initQoI(params);
}


void PhaseField::initField(Mac_input mac){

    for (int i=0; i<2*NUM_PF+1; i++){
        //mac.theta_arr[i+1] = 1.0f*(rand()%10)/(10-1)*(-M_PI/2.0);
       // mac.theta_arr[i+1] = 1.0f*rand()/RAND_MAX*(-M_PI/2.0);
       // mac.theta_arr[i+1] = (i)*grain_gap- M_PI/2.0;
        mac.sint[i] = sinf(mac.theta_arr[i]);
        mac.cost[i] = cosf(mac.theta_arr[i]);
    }  
   
    mac.alpha_mac = new int [(fnx-2*params.ha_wd)*(fny-2*params.ha_wd)];
    read_input(params.mac_folder+"/alpha.txt", mac.alpha_mac);
    printf("%d %d\n", mac.alpha_mac[0], mac.alpha_mac[(fnx-2*params.ha_wd)*(fny-2*params.ha_wd)-1]);

     
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

        alpha[id] = mac.alpha_mac[(j-1)*(fnx-2*params.ha_wd)+(i-1)];
       if (alpha[id]<1 || alpha[id]>NUM_PF) cout<<(j-1)*(fnx-2*params.ha_wd)+(i-1)<<alpha[id]<<endl;
       }

      else {alpha[id]=0;}
      }

    else{
       psi[id]=0.0f;
       phi[id]=0.0f;
     //  Uc[id]=0.0f;
       alpha[id]=0;
 
    }
    }



}


void QOI::initQoI(GlobalConstants params){
    tip_y = new float[num_case*(params.nts+1)];
    frac = new float[num_case*(params.nts+1)*params.num_theta];
    angles = new float[num_case*(2*params.NUM_PF+1)];

    cross_sec = new int[num_case*(params.nts+1)*params.fnx*params.fny];
    alpha = new int[valid_run*params.full_length];
    extra_area = new int[num_case*(params.nts+1)*params.num_theta];
    total_area  = new int[num_case*(params.nts+1)*params.num_theta];
    tip_final   = new int[num_case*(params.nts+1)*params.num_theta];
}



template <typename T>
std::string to_stringp(const T a_value, int n )
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

void h5write_1d(hid_t h5_file, const char* name, void* data, int length, std::string dtype){

    herr_t  status;
    hid_t dataspace, h5data=0;
    hsize_t dim[1];
    dim[0] = length;
    
    dataspace = H5Screate_simple(1, dim, NULL);

    if (dtype.compare("int") ==0){

        h5data = H5Dcreate2(h5_file, name, H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(h5data, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    }
    else if (dtype.compare("float") ==0){
        h5data = H5Dcreate2(h5_file, name, H5T_NATIVE_FLOAT_g, dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(h5data, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    }
    else {

        printf("the data type not specifed");
        status = 1;
    }

    H5Sclose(dataspace);
    H5Dclose(h5data);


}


void PhaseField::output(params_MPI pM){

    string out_format = "ML3D_PF"+to_string(NUM_PF)+"_train"+to_string(q->num_case-q->valid_run)+"_test"+to_string(q->valid_run)+\
    "_Mt"+to_string(params.Mt)+"_grains"+to_string(params.num_theta)+"_frames"+to_string(params.nts)+\
    "_anis"+to_stringp(params.kin_delta,3)+"_G"+to_stringp(params.G,3)+"_Rmax"+to_stringp(params.R,3)+"_seed"+to_string(params.seed_val);
    string out_file = out_format+ "_rank"+to_string(pM.rank)+".h5";
    out_file = "/scratch1/07428/ygqin/graph/" +out_file;
    cout<< "save dir" << out_file <<endl;

    hid_t  h5_file; 


    h5_file = H5Fcreate(out_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    h5write_1d(h5_file, "phi",      phi , length, "float");
    h5write_1d(h5_file, "alpha",  alpha_i_full, full_length, "int");
   // h5write_1d(h5_file, "alpha",    alpha_asse, valid_run*full_length, "int");
  //  h5write_1d(h5_file, "sequence", aseq_asse, num_case*params.num_theta, "int");

    h5write_1d(h5_file, "x_coordinates", x, fnx, "float");
    h5write_1d(h5_file, "y_coordinates", y, fny, "float");
    h5write_1d(h5_file, "z_coordinates", z_full, params.fnz_f, "float");

    h5write_1d(h5_file, "y_t",       q->tip_y,   q->num_case*(params.nts+1), "float");
    h5write_1d(h5_file, "fractions", q->frac,   q->num_case*(params.nts+1)*params.num_theta, "float");
    h5write_1d(h5_file, "angles",    q->angles, q->num_case*(2*NUM_PF+1), "float");

    h5write_1d(h5_file, "extra_area", q->extra_area,   q->num_case*(params.nts+1)*params.num_theta, "int");
    h5write_1d(h5_file, "total_area", q->total_area,   q->num_case*(params.nts+1)*params.num_theta, "int");
    h5write_1d(h5_file, "tip_y_f", q->tip_final,   q->num_case*(params.nts+1)*params.num_theta, "int");
    h5write_1d(h5_file, "cross_sec", q->cross_sec,  q->num_case*(params.nts+1)*fnx*fny, "int");

    H5Fclose(h5_file);


}





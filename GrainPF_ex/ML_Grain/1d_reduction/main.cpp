#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <map>
#include<functional>
#include<mpi.h>
#include "CycleTimer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <hdf5.h>
#include <random>
using namespace std;
#define LS -0.995
#include "include_struct.h"
#define NBW 1
#define NUM_PF 8

void setup( params_MPI pM, GlobalConstants params, Mac_input mac, int fnx, int fny, float* x, float* y, float* phi, float* psi,float* U, int* alpha_i, float* tip_y, float* frac, int* aseq);


// add function for easy retrieving params
template<class T>
T get(std::stringstream& ss) 
{
    T t; 
    ss<<t; 
    if (!ss) // if it failed to read
        throw std::runtime_error("could not parse parameter");
    return t;
}

template <typename T>
std::string to_stringp(const T a_value, int n )
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
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


int h5write_1d(hid_t h5_file, std::string name, float* data, int length, std::string dtype){

	hid_t dataspace, h5data;
	hsize_t dim[1];
	dim[1] = length;
    
    dataspace = H5Screate_simple(1, dim, NULL);

    if (dtype=="int"){

    	h5data = H5Dcreate2(h5_file, name, H5T_NATIVE_INT, dataspacex, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    	status = H5Dwrite(h5data, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    }
    else if (dtype=="float"){
    	h5data = H5Dcreate2(h5_file, name, H5T_NATIVE_FLOAT_g, dataspacex,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    	status = H5Dwrite(h5data, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    }
    else {

    	printf("the data type not specifed");
    	status = 1;
    }

    H5Sclose(dataspace);
    H5Dclose(h5data);

    return status
}


int main(int argc, char** argv)
{
    // All the variables should be float (except for nx, ny, Mt, nts, ictype, which should be integer)

    // step 1 (input): read or calculate  parameters from "input"
    // and print out information: lxd, nx, ny, Mt
    // Create a text string, which is used to output the text file
    params_MPI pM;

//    MPI_Comm comm = MPI_COMM_WORLD;
//    MPI_Init(&argc, &argv);
//    MPI_Comm_rank(comm, &pM.rank);
//    MPI_Comm_size(comm, &pM.nproc);
    pM.rank=0;
    pM.nproc=1;
     
    //if rank == 0: print('GPUs on this node', cuda.gpus)
    //printf('device id',gpu_name,'host id',rank );

    pM.nprocx = (int) ceil(sqrt(pM.nproc));
    pM.nprocy = (int) ceil(pM.nproc/pM.nprocx);


    pM.px = pM.rank%pM.nprocx;           //# x id of current processor   [0:nprocx]
    pM.py = pM.rank/pM.nprocx; //# y id of current processor  [0:nprocy]
    printf("px, %d, py, %d, for rank %d \n",pM.px, pM.py, pM.rank);

    if (pM.rank ==0){ printf("total/x/y processors %d, %d, %d\n", pM.nproc, pM.nprocx, pM.nprocy);}



    char* fileName=argv[1]; 
    std::string mac_folder = argv[2];
    std::string out_direc = "ML_dataset";//argv[2];

    std::string lineText;

    std::ifstream parseFile(fileName);
   // float nx;
   // float Mt;
    int num_case = 220; //1100;
    float grain_size= 2.5;
    bool equal_len = false;
    int valid_run = 20;//100;
    float G0;
    float Rmax;
    float nts;
    float ictype;
    float ha_wd;
    float num_thetaf;
    float temp_Nx, temp_Ny, temp_Nt;
    float seed_val, nprd;
    GlobalConstants params;
    while (parseFile.good()){
        std::getline (parseFile, lineText);
        // Output the text from the file
        getParam(lineText, "G", params.G);
        getParam(lineText, "R", params.R); 
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
        getParam(lineText, "asp_ratio", params.asp_ratio);
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
        getParam(lineText, "beta0", params.beta0);
        // new multiple
        //getParam(lineText, "Ti", params.Ti);
        getParam(lineText, "ha_wd", ha_wd);
        params.ha_wd = (int)ha_wd;
        getParam(lineText, "xmin", params.xmin);
        getParam(lineText, "ymin", params.ymin);
        getParam(lineText, "num_theta", num_thetaf);
        params.num_theta = (int) num_thetaf;
        getParam(lineText, "Nx", temp_Nx);
        getParam(lineText, "Ny", temp_Ny);
        getParam(lineText, "Nt", temp_Nt);
        getParam(lineText, "cfl", params.cfl); 

        getParam(lineText, "Tmelt", params.Tmelt);
        getParam(lineText, "undcool_mean", params.undcool_mean);
        getParam(lineText, "undcool_std", params.undcool_std);
        getParam(lineText, "nuc_Nmax", params.nuc_Nmax);
        getParam(lineText, "nuc_rad", params.nuc_rad);
    }
    
    float dxd = params.dx*params.W0;
    // Close the file
    parseFile.close();
   
    Mac_input mac;
    mac.Nx = (int) temp_Nx;
    mac.Ny = (int) temp_Ny;
    mac.Nt = (int) temp_Nt;
    mac.X_mac = new float[mac.Nx];
    mac.Y_mac = new float[mac.Ny];
    mac.t_mac = new float[mac.Nt];
    mac.alpha_mac = new float [mac.Nx*mac.Ny];
    mac.psi_mac = new float [mac.Nx*mac.Ny];
    mac.U_mac = new float [mac.Nx*mac.Ny];
    mac.T_3D = new float[mac.Nx*mac.Ny*mac.Nt];

  //  std::string mac_folder = "./Takaki/";
    hid_t  h5in_file,  datasetT, dataspaceT, memspace;
    hsize_t dimT[1];
    herr_t  status;
    dimT[0] = mac.Nx*mac.Ny*mac.Nt; 


    read_input(mac_folder+"/x.txt", mac.X_mac);
    read_input(mac_folder+"/y.txt", mac.Y_mac);
    read_input(mac_folder+"/t.txt", mac.t_mac);
   // read_input(mac_folder+"/alpha.txt",mac.alpha_mac);
    read_input(mac_folder+"/psi.txt",mac.psi_mac);
    read_input(mac_folder+"/U.txt",mac.U_mac);
    read_input(mac_folder+"/G.txt", &G0);
    read_input(mac_folder+"/Rmax.txt", &Rmax);
    //G0 = atof(argv[4]);
    //Rmax = atof(argv[5]); 
//    read_input(mac_folder+"/Temp.txt", mac.T_3D);

//    for (int pi = 0; pi<mac.Nt; pi++){
//      mac.t_mac[pi] *= 0.04;     
//    }

    h5in_file = H5Fopen( (mac_folder+"/Temp.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    datasetT = H5Dopen2(h5in_file, "Temp", H5P_DEFAULT);
    dataspaceT = H5Dget_space(datasetT);
    memspace = H5Screate_simple(1,dimT,NULL);
    status = H5Dread(datasetT, H5T_NATIVE_FLOAT, memspace, dataspaceT,
                     H5P_DEFAULT, mac.T_3D);
    printf("mac.T %f\n",mac.T_3D[mac.Nx*mac.Ny*mac.Nt-1]); 

    // calculate the parameters
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
    params.lxd = params.num_theta*grain_size; //-params.xmin; //this has assumption of [,0] params.dx*params.W0*params.nx; # horizontal length in micron
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
    params.ny = (int) (params.asp_ratio*params.nx);
    params.lxd = params.nx*dxd;
    params.lyd = params.ny*dxd;
    params.Mt = (int) (mac.t_mac[mac.Nt-1]/params.tau0/params.dt);
    params.Mt = (params.Mt/2)*2; 
    params.pts_cell = (int) (params.nuc_rad/dxd);

    if (pM.rank==0){ 
    std::cout<<"G0 = "<<G0<<std::endl;
    std::cout<<"Rmax = "<<Rmax<<std::endl;
    std::cout<<"delta = "<<params.delta<<std::endl;
    std::cout<<"kinetic delta = "<<params.kin_delta<<std::endl;
    std::cout<<"beta0 = "<<params.beta0<<std::endl;

    std::cout<<"k = "<<params.k<<std::endl;
    std::cout<<"c_infm = "<<params.c_infm<<std::endl;
   // std::cout<<"Dl = "<<params.Dl<<std::endl;
    std::cout<<"Dh = "<<params.Dh<<std::endl;
    std::cout<<"d0 = "<<params.d0<<std::endl;
    std::cout<<"W0 = "<<params.W0<<std::endl;
    std::cout<<"c_infty = "<<params.c_infty<<std::endl;
    std::cout<<"eps = "<<params.eps<<std::endl;
    std::cout<<"alpha0 = "<<params.alpha0<<std::endl;
    std::cout<<"dx = "<<params.lxd/params.nx/params.W0<<std::endl;
    std::cout<<"dy = "<<params.lyd/params.ny/params.W0<<std::endl;    
    std::cout<<"asp_ratio = "<<params.asp_ratio<<std::endl;
    std::cout<<"nx = "<<params.nx<<std::endl;
    std::cout<<"ny = "<<params.ny<<std::endl;
    //std::cout<<"cfl_coeff = "<<params.cfl<<std::endl;
    std::cout<<"Mt = "<<params.Mt<<std::endl;
    std::cout<<"eta = "<<params.eta<<std::endl;
    std::cout<<"U0 = "<<params.U0<<std::endl;
    std::cout<<"nts = "<<params.nts<<std::endl;
    std::cout<<"ictype = "<<params.ictype<<std::endl;
    //std::cout<<"lT = "<<params.lT<<std::endl;
    std::cout<<"lamd = "<<params.lamd<<std::endl;
    std::cout<<"tau0 = "<<params.tau0<<std::endl;
    //std::cout<<"kinetic effect = "<<params.kin_coeff<<std::endl;
    //std::cout<<"R_tilde = "<<params.R_tilde<<std::endl;
    //std::cout<<"Dl_tilde = "<<params.Dl_tilde<<std::endl;
    std::cout<<"beta0_tilde = "<<params.beta0_tilde<<std::endl;
    std::cout<<"dt = "<<params.dt<<std::endl;
    std::cout<<"lxd = "<<params.lxd<<std::endl;
    std::cout<<"lyd = "<<params.lyd<<std::endl;

    std::cout<<"seed = "<<params.seed_val<<std::endl;
    std::cout<<"noise period = "<<params.noi_period<<std::endl;
    std::cout<<"noise coeff = "<<params.dt_sqrt*params.hi*params.eta<<std::endl;

    std::cout<<"Ti = "<<params.Ti<<std::endl;
    std::cout<<"ha_wd = "<<params.ha_wd<<std::endl;
    std::cout<<"num_grains = "<<params.num_theta<<std::endl;
    std::cout<<"mac Nx = "<<mac.Nx<<std::endl;
    std::cout<<"mac Ny = "<<mac.Ny<<std::endl;
    std::cout<<"mac Nt = "<<mac.Nt<<std::endl;

    std::cout<<"nucleation parameters "<<std::endl;
    std::cout<<"liquidus temperature = "<<params.Tliq<<std::endl;
    std::cout<<"solidus temperature  = "<<params.Tsol<<std::endl;
    std::cout<<"undcooling mean = "<<params.undcool_mean<<std::endl;
    std::cout<<"undcooling std  = "<<params.undcool_std<<std::endl;
    std::cout<<"nucleation density = "<<params.nuc_Nmax<<std::endl;
    std::cout<<"nucleation radius = "<<params.nuc_rad<<std::endl;
    std::cout<<"nucleation points = "<<params.pts_cell<<std::endl;
    }
    // step1 plus: read mat file from macrodata
    //
    //
 
    // step 2 (setup): pass the parameters to constant memory and 
    // allocate and initialize 1_D arrays on CPU/GPU  x: size nx+1, range [0,lxd], y: size ny+1, range [0, lyd]
    // you should get dx = dy = lxd/nx = lyd/ny

    // allocate 1_D arrays on CPU: psi, phi, U of size (nx+3)*(ny+3) -- these are for I/O
    // x and y would be allocate to the shared memory?

    
    
    //==============================
    // parameters depend on MPI
    pM.nx_loc = params.nx/pM.nprocx;
    pM.ny_loc = params.ny/pM.nprocy;
    
    float len_blockx = pM.nx_loc*dxd; //params.lxd/pM.nprocx;
    float len_blocky = pM.ny_loc*dxd; //params.lyd/pM.nprocy;
    
    float xmin_loc = params.xmin+ pM.px*len_blockx; 
    float ymin_loc = params.ymin+ pM.py*len_blocky; 

    if (pM.px==0){pM.nx_loc+=1;}
    else{xmin_loc+=dxd;} //params.lxd/params.nx;}
    
    if (pM.py==0){pM.ny_loc+=1;}
    else{ymin_loc+=dxd;}//params.lyd/params.ny;}

    int length_x = pM.nx_loc+2*params.ha_wd;
    int length_y = pM.ny_loc+2*params.ha_wd;
    float* x=(float*) malloc(length_x* sizeof(float));
    float* y=(float*) malloc(length_y* sizeof(float));


    for(int i=0; i<length_x; i++){
        x[i]=(i-params.ha_wd)*dxd + xmin_loc; 
    }

    for(int i=0; i<length_y; i++){
        y[i]=(i-params.ha_wd)*dxd + ymin_loc;
    }



//===================================

  //  std::cout<<"x= ";
  //  for(int i=0; i<length_x; i++){
  //      std::cout<<x[i]<<" ";
  //  }
    std::cout<< "rank "<< pM.rank<< " xmin " << x[0] <<" xmax "<<x[length_x-1]<<std::endl;

   // std::cout<<"y= ";
   // for(int i=0; i<length_y; i++){
    //    std::cout<<y[i]<<" ";
  //  }
//    std::cout<<std::endl;
        std::cout<< "rank "<< pM.rank<< " ymin "<< y[0] << " ymax "<<y[length_y-1]<<std::endl;

    int length=length_x*length_y;
    std::cout<<"x length of psi, phi, U="<<length_x<<std::endl;
    std::cout<<"y length of psi, phi, U="<<length_y<<std::endl;
    std::cout<<"length of psi, phi, U="<<length<<std::endl;
    float* psi=(float*) malloc(length* sizeof(float));
    float* phi=(float*) malloc(length* sizeof(float));
    float* Uc=(float*) malloc(length* sizeof(float));
    float* alpha=(float*) malloc(length* sizeof(float));    
    int* alpha_i=(int*) malloc(length* sizeof(int));
    float* tip_y=(float*) malloc((params.nts+1)* sizeof(float));
    float* frac=(float*) malloc((params.nts+1)*params.num_theta* sizeof(float));
    int* aseq=(int*) malloc(params.num_theta* sizeof(int));
    float* tip_y_asse=(float*) malloc(num_case*(params.nts+1)* sizeof(float));
    float* frac_asse=(float*) malloc(num_case*(params.nts+1)*params.num_theta* sizeof(float));
    int* aseq_asse=(int*) malloc(num_case*params.num_theta* sizeof(int));
    float* angles_asse=(float*) malloc(num_case*(NUM_PF+1)* sizeof(float));
    int* alpha_asse=(int*) malloc(valid_run*length* sizeof(int));
    //std::cout<<"y= ";
    //for(int i=0+length_y; i<2*length_y; i++){
    //    std::cout<<phi[i]<<" ";
    //}
    //std::cout<<std::endl;



    /*std::cout<<"y= ";
    for(int i=0; i<mac.Nx*mac.Ny*mac.Nt; i++){
        std::cout<<mac.T_3D[i]<<" ";
    }
    std::cout<<std::endl;
   */

    // initialize the angles:
    float grain_gap = M_PI/2.0/(NUM_PF-1);     
    //printf("grain gap %f \n", grain_gap);
    mac.theta_arr = new float[NUM_PF+1];
    mac.cost = new float[NUM_PF+1];
    mac.sint = new float[NUM_PF+1];
    //srand(atoi(argv[3]));
    mac.theta_arr[0] = 0.0f;

    float* frac_ini = (float*) malloc(params.num_theta* sizeof(float));
    float sum_frac = 0.0;
    int* grain_grid = (int*) malloc(params.num_theta* sizeof(int)); 
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(grain_size,0.35*grain_size);

    // start the region of gathering lots of runs
    for (int run=0;run<num_case;run++){
   // for (int run=1005;run<1006;run++){
    printf("case %d\n",run);
    srand(atoi(argv[3])*((int)G0)+run);
    generator.seed( atoi(argv[3])*((int)G0)+run );
   // int* aseq=(int*) malloc(params.num_theta* sizeof(int));
   // initialize the angles for every PF, while keep the liquid 0 
    for (int i=0; i<NUM_PF; i++){
        mac.theta_arr[i+1] = 1.0f*(rand()%10)/(10-1)*(-M_PI/2.0);
       // mac.theta_arr[i+1] = (i)*grain_gap- M_PI/2.0;
        mac.sint[i+1] = sinf(mac.theta_arr[i+1]);
        mac.cost[i+1] = cosf(mac.theta_arr[i+1]);
    }  
   
    sum_frac = 0.0f;
    for (int i=0; i<params.num_theta; i++){
       aseq[i] = i+1; //rand()%NUM_PF +1;
       frac_ini[i] = abs(distribution(generator)); 
       sum_frac += frac_ini[i];
    }
   
    for (int i=0; i<params.num_theta; i++){
        frac_ini[i] /= sum_frac;
        grain_grid[i] = (int) (frac_ini[i]*pM.nx_loc);
    printf("grain %d, PF %d, angle %f, grid %d, frac %f\n",i,aseq[i],mac.theta_arr[i+1]/M_PI*180,grain_grid[i],frac_ini[i]);
    }

    for (int i=0; i<params.num_theta-1; i++){
        grain_grid[params.num_theta-1] = pM.nx_loc - grain_grid[i];
    }

    float Dx = mac.X_mac[mac.Nx-1] - mac.X_mac[mac.Nx-2];
    float Dy = mac.Y_mac[mac.Ny-1] - mac.Y_mac[mac.Ny-2];
    for(int id=0; id<length; id++){

      int j = id/length_x;
      int i = id%length_x; 
      
      if ( (i>params.ha_wd-1) && (i<length_x-params.ha_wd) && (j>params.ha_wd-1) && (j<length_y-params.ha_wd) ){
      int kx = (int) (( x[i] - mac.X_mac[0] )/Dx);
      int rkx = (int) round(( x[i] - mac.X_mac[0] )/Dx);
      float delta_x = ( x[i] - mac.X_mac[0] )/Dx - kx;
         //printf("%f ",delta_x);
      int ky = (int) (( y[j] - mac.Y_mac[0] )/Dy);
      int rky = (int) round(( y[j] - mac.Y_mac[0] )/Dy);
      float delta_y = ( y[j] - mac.Y_mac[0] )/Dy - ky;
      if (kx==mac.Nx-1) {kx = mac.Nx-2; delta_x =1.0f;}
      if (ky==mac.Ny-1) {ky = mac.Ny-2; delta_y =1.0f;}
      int offset = kx + ky*mac.Nx;
      int roffset = rkx + rky*mac.Nx;
      if (offset>mac.Nx*mac.Ny-1-1-mac.Nx) printf("%d, %d, %d, %d  ", i,j,kx,ky);
      psi[id] =  (1.0f-delta_x)*(1.0f-delta_y)*mac.psi_mac[ offset ] + (1.0f-delta_x)*delta_y*mac.psi_mac[ offset+mac.Nx ]+\
               delta_x*(1.0f-delta_y)*mac.psi_mac[ offset+1 ] +   delta_x*delta_y*mac.psi_mac[ offset+mac.Nx+1 ];

      Uc[id] =  (1.0f-delta_x)*(1.0f-delta_y)*mac.U_mac[ offset ] + (1.0f-delta_x)*delta_y*mac.U_mac[ offset+mac.Nx ]+\
               delta_x*(1.0f-delta_y)*mac.U_mac[ offset+1 ] +   delta_x*delta_y*mac.U_mac[ offset+mac.Nx+1 ];
  
     //   psi[id]=0.0;
      psi[id] = psi[id]/params.W0;
      phi[id]=tanhf(psi[id]/params.sqrt2);
     //   Uc[id]=0.0;
      if (phi[id]>LS){

      int aid=0;
      if (equal_len==true){
      aid = (int) (x[i]/(params.lxd/params.num_theta));
      if (aid==params.num_theta) {aid = params.num_theta-1;}}
      else{
      int left_bc = 0; 
      int right_bc = grain_grid[0]; 
      for (int gr_id=0;gr_id<params.num_theta;gr_id++){
           if ( (left_bc<i) && (i<=right_bc) ) {aid=gr_id;break;}
           else {left_bc+=grain_grid[gr_id];right_bc+=grain_grid[gr_id+1];}
      }
      }
      alpha_i[id] = aseq[aid];
      if ( (alpha_i[id]>=0) || (alpha_i[id]<=params.num_theta-1) ){}
      else {printf("alpha is wrong \n");exit(1);}

       }

      else {alpha_i[id]=0;}
      }

    else{
       psi[id]=0.0f;
       phi[id]=0.0f;
       Uc[id]=0.0f;
       alpha_i[id]=0;
 
    }
    }


    setup( pM, params, mac, length_x, length_y, x, y, phi, psi, Uc, alpha_i, tip_y, frac, aseq);

    // save the QoIs 
    //float* tip_y_asse=(float*) malloc(num_case*(params.nts+1)* sizeof(float));
    //float* frac_asse=(float*) malloc(num_case*(params.nts+1)*params.num_theta* sizeof(float));
    //int* aseq_asse=(int*) malloc(num_case*params.num_theta* sizeof(int));
    memcpy(tip_y_asse+run*(params.nts+1),tip_y,sizeof(float)*(params.nts+1));  
    memcpy(frac_asse+run*(params.nts+1)*params.num_theta,frac,sizeof(float)*(params.nts+1)*params.num_theta); 
    memcpy(aseq_asse+run*params.num_theta,aseq,sizeof(int)*params.num_theta);
    memcpy(angles_asse+run*(NUM_PF+1),mac.theta_arr,sizeof(float)*(NUM_PF+1));

    if (run>=num_case-valid_run){
        int loca_case = run-(num_case-valid_run);
           memcpy(alpha_asse+loca_case*length,alpha_i,sizeof(int)*length);

    }   

    }
    // U to c
    float cinf_cl0 =  1.0f+ (1.0f-params.k)*params.U0;
    for(int id=0; id<length; id++){

      Uc[id] = ( 1.0f+ (1.0f-params.k)*Uc[id] )*( params.k*(1.0f+phi[id])/2.0f + (1.0f-phi[id])/2.0f ) / cinf_cl0;
      if (phi[id]>LS){alpha[id]=mac.theta_arr[alpha_i[id]]+M_PI/2.0;}
      else {alpha[id]=0.0f;} 
    }


    //std::cout<<"y= ";
    //for(int i=0+length_y; i<2*length_y; i++){
    //    std::cout<<Uc[i]<<" ";
    //}
    //std::cout<<std::endl;
    // step 3 (time marching): call the kernels Mt times
    int phs = NUM_PF;
    string out_format = "ML_PF"+to_string(phs)+"_train"+to_string(num_case-valid_run)+"_test"+to_string(valid_run)+"_Mt"+to_string(params.Mt)+"_grains"+to_string(params.num_theta)+"_frames"+to_string(params.nts)+"_anis"+to_stringp(params.kin_delta,3)+"_G0"+to_stringp(G0,3)+"_Rmax"+to_stringp(Rmax,3)+"_seed"+to_string(atoi(argv[3]));
    string out_file = out_format+ "_rank"+to_string(pM.rank)+".h5";
    out_file = "/scratch/07428/ygqin/Aeolus/Fast_code/" + out_direc + "/" +out_file;
   // ofstream out( out_file );
   // out.precision(5);
   // copy( phi, phi + length, ostream_iterator<float>( out, "\n" ) );

    // claim file and dataset handles
    hid_t  h5_file; 
    /*alpha_o, yt_o, frac_o, seq_o, angle_o, dataspace, xcoor, ycoor, dataspacex, dataspacey, dataspacet, dataspacefrac, dataspaceg, dataspaceang;
    hsize_t dimx[1], dimy[1], dimf[1], dimg[1], dimfrac[1], dimnt[1], dimang[1];
    dimx[0]=length_x; dimy[0]=length_y; dimf[0]=valid_run*length;
    dimg[0] = num_case*params.num_theta; 
    dimfrac[0] = num_case*(params.nts+1)*params.num_theta; 
    dimnt[0] = num_case*(params.nts+1);
    dimang[0] = num_case*(NUM_PF+1);
    */

    h5_file = H5Fcreate(out_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);


    h5write_1d(h5_file, "alpha",    alpha_asse, valid_run*length, "int");
    h5write_1d(h5_file, "sequence", aseq_asse, num_case*params.num_theta, "int");

    h5write_1d(h5_file, "x_coordinates", x, length_x, "float");
    h5write_1d(h5_file, "y_coordinates", y, length_y, "float");

    h5write_1d(h5_file, "y_t",       tip_y_asse,   num_case*(params.nts+1), "float");
    h5write_1d(h5_file, "fractions", frac_asse,   num_case*(params.nts+1)*params.num_theta, "float");
    h5write_1d(h5_file, "angles",    angles_asse, num_case*(NUM_PF+1), "float");


    /*
    dataspace = H5Screate_simple(1, dimf, NULL);
    dataspacex = H5Screate_simple(1, dimx, NULL);
    dataspacey = H5Screate_simple(1, dimy, NULL);
    dataspacet =  H5Screate_simple(1, dimnt, NULL);
    dataspacefrac = H5Screate_simple(1, dimfrac, NULL);
    dataspaceg =  H5Screate_simple(1, dimg, NULL);
    dataspaceang = H5Screate_simple(1, dimang, NULL);

    xcoor = H5Dcreate2(h5_file, "x_coordinates", H5T_NATIVE_FLOAT_g, dataspacex,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ycoor = H5Dcreate2(h5_file, "y_coordinates", H5T_NATIVE_FLOAT_g, dataspacey,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    alpha_o = H5Dcreate2(h5_file, "alpha", H5T_NATIVE_INT, dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    yt_o = H5Dcreate2(h5_file, "y_t", H5T_NATIVE_FLOAT_g, dataspacet,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    frac_o = H5Dcreate2(h5_file, "fractions", H5T_NATIVE_FLOAT_g, dataspacefrac,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    seq_o = H5Dcreate2(h5_file, "sequence", H5T_NATIVE_INT, dataspaceg,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    angle_o = H5Dcreate2(h5_file, "angles", H5T_NATIVE_FLOAT_g, dataspaceang,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite(alpha_o, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, alpha_asse);
    status = H5Dwrite(xcoor, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
    status = H5Dwrite(ycoor, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, y);
    status = H5Dwrite(frac_o, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, frac_asse);
    status = H5Dwrite(yt_o, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, tip_y_asse);
    status = H5Dwrite(seq_o, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, aseq_asse);
    status = H5Dwrite(angle_o, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, angles_asse);
    // close all the hdf handles
    H5Sclose(dataspace);H5Sclose(dataspacex);
    H5Dclose(alpha_o);
    H5Dclose(xcoor);
    H5Dclose(ycoor);H5Sclose(dataspacey);
	*/

    H5Fclose(h5_file);
    H5Dclose(datasetT);
    H5Sclose(dataspaceT);
    H5Sclose(memspace);
    H5Fclose(h5in_file);

    //MPI_Finalize();
    delete[] phi;
    delete[] Uc;
    delete[] psi;
    return 0;
}

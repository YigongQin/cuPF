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
#include "params.h"
#include <random>
#include "PhaseField.h"
using namespace std;


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
    //argv[2];

    std::string lineText;

    std::ifstream parseFile(fileName);


    float G0;
    float Rmax;
    float nts;
    float ictype;
    float ha_wd;
    float num_thetaf;
    float temp_Nx, temp_Ny, temp_Nz, temp_Nt;
    float seed_val, nprd;
    GlobalConstants params;
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
   
    Mac_input mac;
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
    
  //  std::string mac_folder = "./Takaki/";
    hid_t  h5in_file,  datasetT, dataspaceT, memspace;
    hsize_t dimT[1];
    herr_t  status;
    dimT[0] = mac.Nx*mac.Ny*mac.Nz*mac.Nt; 


    read_input(mac_folder+"/x.txt", mac.X_mac);
    read_input(mac_folder+"/y.txt", mac.Y_mac);
    read_input(mac_folder+"/z.txt", mac.Z_mac);
    read_input(mac_folder+"/t.txt", mac.t_mac);
   // read_input(mac_folder+"/alpha.txt",mac.alpha_mac);
    read_input(mac_folder+"/psi.txt",mac.psi_mac);
    read_input(mac_folder+"/U.txt",mac.U_mac);
    read_input(mac_folder+"/G.txt", &G0);
    read_input(mac_folder+"/Rmax.txt", &Rmax);
    read_input(mac_folder+"/NG.txt", &num_thetaf);
    params.num_theta = (int) num_thetaf;
    params.NUM_PF = params.num_theta;
    int NUM_PF = params.NUM_PF;
    mac.theta_arr = new float[2*NUM_PF+1];
    read_input(mac_folder+"/theta.txt", mac.theta_arr);

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
    printf("mac.T %f\n",mac.T_3D[mac.Nx*mac.Ny*mac.Nz*mac.Nt-1]); 

    // calculate the parameters
    int grain_dim = (int) sqrt(params.num_theta);
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
    std::cout<<"dz = "<<params.lyd/params.ny/params.W0<<std::endl;   
    std::cout<<"asp_ratio_yx = "<<params.asp_ratio_yx<<std::endl;
    std::cout<<"asp_ratio_zx = "<<params.asp_ratio_zx<<std::endl;
    std::cout<<"nx = "<<params.nx<<std::endl;
    std::cout<<"ny = "<<params.ny<<std::endl;
    std::cout<<"nz = "<<params.nz<<std::endl;
    std::cout<<"full nz = "<<params.nz_full<<std::endl;
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
    std::cout<<"mac Nz = "<<mac.Nz<<std::endl;
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

    params.seed_val = atoi(argv[3]);

    PhaseField* pf_solver; // initialize the pointer to the class
    pf_solver = new PhaseField();
    pf_solver->params = params; //set parameters of simulations
    pf_solver->q->num_case = 1;  //set parameters of realizations
    pf_solver->q->valid_run = 1; 
    pf_solver->cpuSetup(pM);

 //   int fnx = pf_solver->params.fnx, fny = pf_solver->params.fny, fnz = pf_solver->params.fnz, \
 //   length = pf_solver->params.length ,full_length = pf_solver->params.full_length;

    
//===================================



    mac.alpha_mac = new int [pM.nx_loc*pM.ny_loc];
    read_input(mac_folder+"/alpha.txt", mac.alpha_mac);
    printf("%d %d\n", mac.alpha_mac[0], mac.alpha_mac[pM.nx_loc*pM.ny_loc-1]);


    

   // mac.theta_arr = new float[2*NUM_PF+1];
    mac.cost = new float[2*NUM_PF+1];
    mac.sint = new float[2*NUM_PF+1];
    //srand(atoi(argv[3]));
    mac.theta_arr[0] = 0.0f;

 //   float* frac_ini = (float*) malloc(params.num_theta* sizeof(float));
 //   float sum_frac = 0.0;
 //   int* grain_grid = (int*) malloc(params.num_theta* sizeof(int)); 
  //  std::default_random_engine generator;
 //   std::normal_distribution<float> distribution(grain_size,0.35*grain_size);

    // start the region of gathering lots of runs
    for (int run=0;run<pf_solver->q->num_case;run++){
   
    printf("case %d\n",run);
   // int loc_seed = 20*((int)G0) + (int) (10000*Rmax);
   // loc_seed = loc_seed*num_case + run;
   // srand(loc_seed);
   // generator.seed( loc_seed );
   // int* aseq=(int*) malloc(params.num_theta* sizeof(int));
   // initialize the angles for every PF, while keep the liquid 0 
   // for (int i=0; i<params.num_theta; i++){
    //   aseq[i] = i+1;} //rand()%NUM_PF +1;

    pf_solver->initField(mac);
    pf_solver->cudaSetup(pM);
    pf_solver->evolve(mac);

  //  if (run>=num_case-valid_run){
   //     int loca_case = run-(num_case-valid_run);
    //       memcpy(alpha_asse+loca_case*full_length, pf_solver->alpha_i_full,sizeof(int)*full_length);

   // }   

    }



    pf_solver->output(pM);
    H5Dclose(datasetT);
    H5Sclose(dataspaceT);
    H5Sclose(memspace);
    H5Fclose(h5in_file);

    //MPI_Finalize();

    return 0;
}

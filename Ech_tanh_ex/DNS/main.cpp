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
using namespace std;
#define LS -0.995
#include "include_struct.h"



void setup(MPI_Comm comm, params_MPI pM, GlobalConstants params, Mac_input mac, int fnx, int fny, float* x, float* y, float* phi, float* psi,float* U, float* alpha);


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
/*
void matread(std::string matfile, std::string key, std::vector<double>& v)
{
    // open MAT-file
    MATFile *pmat = matOpen(matfile, "r");
    if (pmat == NULL) return;

    // extract the specified variable
    mxArray *arr = matGetVariable(pmat, key);
    if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {
        // copy data
        mwSize num = mxGetNumberOfElements(arr);
        double *pr = mxGetPr(arr);
        if (pr != NULL) {
            v.reserve(num); //is faster than resize :-)
            v.assign(pr, pr+num);
        }
    }

    // cleanup
    mxDestroyArray(arr);
    matClose(pmat);
}
*/


int main(int argc, char** argv)
{
    // All the variables should be float (except for nx, ny, Mt, nts, ictype, which should be integer)

    // step 1 (input): read or calculate  parameters from "input"
    // and print out information: lxd, nx, ny, Mt
    // Create a text string, which is used to output the text file
    params_MPI pM;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &pM.rank);
    MPI_Comm_size(comm, &pM.nproc);


     
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
    std::string out_direc = argv[2];

    std::string lineText;

    std::ifstream parseFile(fileName);
   // float nx;
   // float Mt;
    float nts;
    float ictype;
    float ha_wd;
    float num_theta;
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
        getParam(lineText, "c_infm", params.c_infm); 
        getParam(lineText, "Dl", params.Dl); 
        getParam(lineText, "d0", params.d0); 
        getParam(lineText, "W0", params.W0);  
        getParam(lineText, "c_infty", params.c_infty);
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

        // new multiple
        getParam(lineText, "Ti", params.Ti);
        getParam(lineText, "ha_wd", ha_wd);
        params.ha_wd = (int)ha_wd;
        getParam(lineText, "xmin", params.xmin);
        getParam(lineText, "ymin", params.ymin);
        getParam(lineText, "num_theta", num_theta);

        getParam(lineText, "Nx", temp_Nx);
        getParam(lineText, "Ny", temp_Ny);
        getParam(lineText, "Nt", temp_Nt); 
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
    read_input(mac_folder+"/alpha.txt",mac.alpha_mac);
    read_input(mac_folder+"/psi.txt",mac.psi_mac);
    read_input(mac_folder+"/U.txt",mac.U_mac);
//    read_input(mac_folder+"/Temp.txt", mac.T_3D);

    h5in_file = H5Fopen( (mac_folder+"/Temp.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    datasetT = H5Dopen2(h5in_file, "Temp", H5P_DEFAULT);
    dataspaceT = H5Dget_space(datasetT);
    memspace = H5Screate_simple(1,dimT,NULL);
    status = H5Dread(datasetT, H5T_NATIVE_FLOAT, memspace, dataspaceT,
                     H5P_DEFAULT, mac.T_3D);
    printf("mac.T %f\n",mac.T_3D[mac.Nx*mac.Ny*mac.Nt-1]); 
    // calculate the parameters
    params.lT = params.c_infm*( 1.0/params.k-1 )/params.G;//       # thermal length           um
    params.lamd = 5*sqrt(2.0)/8*params.W0/params.d0;//     # coupling constant
    params.tau0 = 0.6267*params.lamd*params.W0*params.W0/params.Dl; //    # time scale  
    params.R_tilde = params.R*params.tau0/params.W0;
    params.Dl_tilde = params.Dl*params.tau0/pow(params.W0,2);
    params.lT_tilde = params.lT/params.W0;
    params.dt = 0.25*0.8*pow(params.dx,2)/(4*params.Dl_tilde);
//    params.ny = (int) (params.asp_ratio*params.nx);
    params.lxd = -params.xmin*0.99; //this has assumption of [,0] params.dx*params.W0*params.nx; # horizontal length in micron
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

    if (pM.rank==0){ 
    std::cout<<"G = "<<params.G<<std::endl;
    std::cout<<"R = "<<params.R<<std::endl;
    std::cout<<"delta = "<<params.delta<<std::endl;
    std::cout<<"k = "<<params.k<<std::endl;
    std::cout<<"c_infm = "<<params.c_infm<<std::endl;
    std::cout<<"Dl = "<<params.Dl<<std::endl;
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
    std::cout<<"Mt = "<<params.Mt<<std::endl;
    std::cout<<"eta = "<<params.eta<<std::endl;
    std::cout<<"U0 = "<<params.U0<<std::endl;
    std::cout<<"nts = "<<params.nts<<std::endl;
    std::cout<<"ictype = "<<params.ictype<<std::endl;
    std::cout<<"lT = "<<params.lT<<std::endl;
    std::cout<<"lamd = "<<params.lamd<<std::endl;
    std::cout<<"tau0 = "<<params.tau0<<std::endl;
    std::cout<<"R_tilde = "<<params.R_tilde<<std::endl;
    std::cout<<"Dl_tilde = "<<params.Dl_tilde<<std::endl;
    std::cout<<"lT_tilde = "<<params.lT_tilde<<std::endl;
    std::cout<<"dt = "<<params.dt<<std::endl;
    std::cout<<"lxd = "<<params.lxd<<std::endl;
    std::cout<<"lyd = "<<params.lyd<<std::endl;

    std::cout<<"seed = "<<params.seed_val<<std::endl;
    std::cout<<"noise period = "<<params.noi_period<<std::endl;
    std::cout<<"noise coeff = "<<params.dt_sqrt*params.hi*params.eta<<std::endl;

    std::cout<<"Ti = "<<params.Ti<<std::endl;
    std::cout<<"ha_wd = "<<params.ha_wd<<std::endl;
    std::cout<<"num_theta = "<<num_theta<<std::endl;
    std::cout<<"mac Nx = "<<mac.Nx<<std::endl;
    std::cout<<"mac Ny = "<<mac.Ny<<std::endl;
    std::cout<<"mac Nt = "<<mac.Nt<<std::endl;
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
    float grain_gap = M_PI/2.0/num_theta;     
    printf("grain gap %f \n", grain_gap);
    float* theta_arr=(float*) malloc(num_theta* sizeof(float));
    for (int i=0; i<num_theta; i++){
        theta_arr[i] = 1.0f*rand()/RAND_MAX*(-M_PI/2.0);
    }

    float Dx = mac.X_mac[1] - mac.X_mac[0];
    float Dy = mac.Y_mac[1] - mac.Y_mac[0];
    for(int id=0; id<length; id++){

      int j = id/length_x;
      int i = id%length_x; 
      
      if ( (i>params.ha_wd-1) && (i<length_x-params.ha_wd) && (j>params.ha_wd-1) && (j<length_y-params.ha_wd) ){
      int kx = (int) (( x[i] - mac.X_mac[0] )/Dx);
      float delta_x = ( x[i] - mac.X_mac[0] )/Dx - kx;
         //printf("%f ",delta_x);
      int ky = (int) (( y[j] - mac.Y_mac[0] )/Dy);
      float delta_y = ( y[j] - mac.Y_mac[0] )/Dy - ky;
      int offset = kx + ky*mac.Nx;
     // if (offset>mac.Nx*mac.Ny-1-1-length_x) printf("%d, %d  ", i,j);
      psi[id] =  (1.0f-delta_x)*(1.0f-delta_y)*mac.psi_mac[ offset ] + (1.0f-delta_x)*delta_y*mac.psi_mac[ offset+mac.Nx ]+\
               delta_x*(1.0f-delta_y)*mac.psi_mac[ offset+1 ] +   delta_x*delta_y*mac.psi_mac[ offset+mac.Nx+1 ];

      Uc[id] =  (1.0f-delta_x)*(1.0f-delta_y)*mac.U_mac[ offset ] + (1.0f-delta_x)*delta_y*mac.U_mac[ offset+mac.Nx ]+\
               delta_x*(1.0f-delta_y)*mac.U_mac[ offset+1 ] +   delta_x*delta_y*mac.U_mac[ offset+mac.Nx+1 ];
  
     //   psi[id]=0.0;
      psi[id] = psi[id]/params.W0;
      phi[id]=tanhf(psi[id]/params.sqrt2);
     //   Uc[id]=0.0;
      if (phi[id]>LS){
      alpha[id] =  (1.0f-delta_x)*(1.0f-delta_y)*mac.alpha_mac[ offset ] + (1.0f-delta_x)*delta_y*mac.alpha_mac[ offset+mac.Nx ]\
               +delta_x*(1.0f-delta_y)*mac.alpha_mac[ offset+1 ] +   delta_x*delta_y*mac.alpha_mac[ offset+mac.Nx+1 ];
     // if (alpha[id]<-1.1) printf("%f ",alpha[id]); 
      int theta_id = (int) ( (alpha[id]+M_PI/2.0)/grain_gap);
      if (theta_id>=num_theta) printf("theta overflow \n");
      alpha[id] = theta_id*grain_gap-M_PI/2.0; // 
     // alpha[id] = theta_arr[theta_id];
       }

      else {alpha[id]=0.0f;}
      }

    else{
       psi[id]=0.0f;
       phi[id]=0.0f;
       Uc[id]=0.0f;
       alpha[id]=0.0f;
 
    }
    }


    setup(comm, pM, params, mac, length_x, length_y, x, y, phi, psi, Uc, alpha);



    // U to c
    float cinf_cl0 =  1.0f+ (1.0f-params.k)*params.U0;
    for(int id=0; id<length; id++){

      Uc[id] = ( 1.0f+ (1.0f-params.k)*Uc[id] )*( params.k*(1.0f+phi[id])/2.0f + (1.0f-phi[id])/2.0f ) / cinf_cl0;
      if (phi[id]>LS){alpha[id]+=M_PI/2.0;}
      else {alpha[id]=0.0f;} 
    }


    //std::cout<<"y= ";
    //for(int i=0+length_y; i<2*length_y; i++){
    //    std::cout<<Uc[i]<<" ";
    //}
    //std::cout<<std::endl;
    // step 3 (time marching): call the kernels Mt times
    string out_format = "DNS_nx"+to_string(params.nx)+"_ny"+to_string(params.ny)+"_seed"+to_string(params.seed_val);
    string out_file = out_format+ "_rank"+to_string(pM.rank)+".h5";
    out_file = "/scratch/07428/ygqin/Aeolus/Fast_code/" + out_direc + "/" +out_file; 
   // ofstream out( out_file );
   // out.precision(5);
   // copy( phi, phi + length, ostream_iterator<float>( out, "\n" ) );

    // claim file and dataset handles
    hid_t  h5_file, phi_o, U_o, alpha_o, dataspace, xcoor, ycoor, dataspacex, dataspacey;
    hsize_t dimsf[2], dimx[1], dimy[1], dimf[1];
    dimsf[0] = length; dimsf[1] = params.nts; dimx[0]=length_x; dimy[0]=length_y; dimf[0]=length;

    // claim file and dataset handles
    // create file, data spaces and datasets
    h5_file = H5Fcreate(out_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    dataspace = H5Screate_simple(1, dimf, NULL);
    dataspacex = H5Screate_simple(1, dimx, NULL);
    dataspacey = H5Screate_simple(1, dimy, NULL);
    xcoor = H5Dcreate2(h5_file, "x_coordinates", H5T_NATIVE_FLOAT_g, dataspacex,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ycoor = H5Dcreate2(h5_file, "y_coordinates", H5T_NATIVE_FLOAT_g, dataspacey,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    phi_o = H5Dcreate2(h5_file, "phi", H5T_NATIVE_FLOAT_g, dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    U_o = H5Dcreate2(h5_file, "Uc", H5T_NATIVE_FLOAT_g, dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    alpha_o = H5Dcreate2(h5_file, "alpha", H5T_NATIVE_FLOAT_g, dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // write the coordinates, temperature field and the true solution to the hdf5 file
    status = H5Dwrite(phi_o, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, phi);
    status = H5Dwrite(U_o, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, Uc);
    status = H5Dwrite(alpha_o, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, alpha);
    status = H5Dwrite(xcoor, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
    status = H5Dwrite(ycoor, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, y);

    // close all the hdf handles
    H5Sclose(dataspace);H5Sclose(dataspacex);
    H5Dclose(phi_o);
    H5Dclose(U_o);H5Dclose(alpha_o);
    H5Dclose(xcoor);
    H5Dclose(ycoor);H5Sclose(dataspacey);
    H5Fclose(h5_file);
    H5Dclose(datasetT);
    H5Sclose(dataspaceT);
    H5Sclose(memspace);
    H5Fclose(h5in_file);
    MPI_Finalize();
    delete[] phi;
    delete[] Uc;
    delete[] psi;
    return 0;
}

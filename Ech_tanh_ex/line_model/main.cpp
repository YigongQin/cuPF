#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <map>
#include<functional>

#include "CycleTimer.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <hdf5.h>
// #include <mat.h> 
using namespace std;


void printCudaInfo();

struct GlobalConstants {
  int nx;
  int ny;
  int Mt;
  int nts; 
  int ictype;
  float G;
  float R;
  float delta;
  float k;
  float c_infm;
  float Dl;
  float d0;
  float W0;
  float lT;
  float lamd; 
  float tau0;
  float c_infty; 
  float R_tilde;
  float Dl_tilde; 
  float lT_tilde; 
  float eps; 
  float alpha0; 
  float dx; 
  float dt; 
  float asp_ratio; 
  float lxd;
  float lx; 
  float lyd; 
  float eta; 
  float U0; 
  int seed_val;
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
};


struct Mac_input{
  int Ny;
  int Nt;
  float* t_mac;
  float* y_mac;
  float* Gt; 
  float* Rt; 
  float* psi_mac;
  float* U_mac;

};


void setup(Mac_input mac, GlobalConstants params, int fnx, int fny, float* x, float* y, float* phi, float* psi,float* U);


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


int main(int argc, char** argv)
{
    // All the variables should be float (except for nx, ny, Mt, nts, ictype, which should be integer)

    // step 1 (input): read or calculate  parameters from "input"
    // and print out information: lxd, nx, ny, Mt
    // Create a text string, which is used to output the text file
    char* fileName=argv[1];
    std::string mac_folder = argv[2];
    std::string out_direc = argv[2];
    std::string lineText;

    std::ifstream parseFile(fileName);
   // float nx;
   // float Mt;
    float nts;
    float ictype;
    float seed_val;
    float nprd;
    float temp_Ny, temp_Nt;
    float ztip0;
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
        getParam(lineText, "lxd", params.lxd);
        //params.nx = (int)nx;
     //   getParam(lineText, "Mt", Mt);
       // params.Mt = (int)Mt;
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
    
        getParam(lineText, "Ny", temp_Ny);
        getParam(lineText, "Nt", temp_Nt);
        getParam(lineText, "ztip0", ztip0);
    }

    // Close the file
    parseFile.close();
    float dxd = params.dx*params.W0;
    Mac_input mac;
    mac.Ny = (int) temp_Ny;
    mac.Nt = (int) temp_Nt;
    mac.t_mac = new float[mac.Nt];
    mac.Gt = new float[mac.Nt];
    mac.Rt = new float[mac.Nt];
    mac.psi_mac = new float [mac.Ny];
    mac.U_mac = new float [mac.Ny];
    mac.y_mac = new float [mac.Ny];   


    read_input(mac_folder+"/Gt.txt", mac.Gt);
    read_input(mac_folder+"/Rt.txt", mac.Rt);
    read_input(mac_folder+"/t.txt", mac.t_mac);
    read_input(mac_folder+"/y.txt",mac.y_mac);
    read_input(mac_folder+"/psi.txt",mac.psi_mac);
    read_input(mac_folder+"/U.txt",mac.U_mac);
 
    // calculate the parameters
    params.lT = params.c_infm*( 1.0/params.k-1 )/params.G;//       # thermal length           um
    params.lamd = 5*sqrt(2.0)/8*params.W0/params.d0;//     # coupling constant
    params.tau0 = 0.6267*params.lamd*params.W0*params.W0/params.Dl; //    # time scale  
    params.R_tilde = params.R*params.tau0/params.W0;
    params.Dl_tilde = params.Dl*params.tau0/pow(params.W0,2);
    params.lT_tilde = params.lT/params.W0;
    params.dt = 0.8*pow(params.dx,2)/(4*params.Dl_tilde); //0.0142639;
    params.nx = (int) (params.lxd/dxd);
    params.ny = (int) (params.asp_ratio*params.nx);
    params.Mt = (int) (mac.t_mac[mac.Nt-1]/params.tau0/params.dt);
    params.Mt = (params.Mt/2)*2;
//    params.lxd = params.dx*params.W0*params.nx; //                    # horizontal length in micron
    params.lyd = params.asp_ratio*params.lxd;
    params.hi = 1.0/params.dx;
    params.cosa = cos(params.alpha0/180*M_PI);
    params.sina = sin(params.alpha0/180*M_PI);
    params.sqrt2 = sqrt(2.0);
    params.a_s = 1 - 3.0*params.delta;
    params.epsilon = 4.0*params.delta/params.a_s;
    params.a_12 = 4.0*params.a_s*params.epsilon;
    params.dt_sqrt = sqrt(params.dt); 
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
    std::cout<<"dx = "<<params.dx<<std::endl;
    std::cout<<"asp_ratio = "<<params.asp_ratio<<std::endl;
    std::cout<<"nx = "<<params.nx<<std::endl;
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
    std::cout<<"ny = "<<params.ny<<std::endl;
    std::cout<<"lxd = "<<params.lxd<<std::endl;
    std::cout<<"lyd = "<<params.lyd<<std::endl;
    std::cout<<"noise coeff = "<<params.dt_sqrt*params.hi*params.eta<<std::endl; 
    std::cout<<"noise seed"<<params.seed_val<<std::endl;
    // step 2 (setup): pass the parameters to constant memory and 
    // allocate and initialize 1_D arrays on CPU/GPU  x: size nx+1, range [0,lxd], y: size ny+1, range [0, lyd]
    // you should get dx = dy = lxd/nx = lyd/ny

    // allocate 1_D arrays on CPU: psi, phi, U of size (nx+3)*(ny+3) -- these are for I/O
    // x and y would be allocate to the shared memory?
    
    int length_x = params.nx+2;
    int length_y = params.ny+3;
    float* x=(float*) malloc(length_x* sizeof(float));
    float* y=(float*) malloc(length_y* sizeof(float));

    // float* x = NULL;
    // float* y = NULL;
    // cudaMallocManaged((void**)&x, length_x* sizeof(float));
    // cudaMallocManaged((void**)&y, length_y* sizeof(float));

    // x
    for(int i=0; i<length_x; i++){
        x[i]=(i-1)*dxd; //params.lxd/params.nx; 
    }

    //std::cout<<"x= ";
    //for(int i=0; i<length_x; i++){
    //    std::cout<<x[i]<<" ";
   // }
    //std::cout<<std::endl;

    // y
    for(int i=0; i<length_y; i++){
        y[i]=(i-1)*dxd+mac.y_mac[0]; //params.lyd/params.ny; 
    }
        std::cout<< " ymin "<< y[0] << " ymax "<<y[length_y-1]<<std::endl;
    //std::cout<<"y= ";
    //for(int i=0; i<length_y; i++){
    //    std::cout<<y[i]<<" ";
    //}
    //std::cout<<std::endl;

    int length=length_x*length_y;
    std::cout<<"length of psi, phi, U="<<length<<std::endl;
    float* psi=(float*) malloc(length* sizeof(float));
    float* phi=(float*) malloc(length* sizeof(float));
    float* Uc=(float*) malloc(length* sizeof(float));

    float Dy = mac.y_mac[1] - mac.y_mac[0];
    for(int id=0; id<length; id++){

      int j = id/length_x;
      int i = id%length_x;

      if ( (i>0) && (i<length_x-1) && (j>0) && (j<length_y-1) ){
      int ky = (int) (( y[j] - mac.y_mac[0] )/Dy);
      float delta_y = ( y[j] - mac.y_mac[0] )/Dy - ky;
     // if (offset>mac.Nx*mac.Ny-1-1-length_x) printf("%d, %d  ", i,j);
      psi[id] =  (1.0f-delta_y)*mac.psi_mac[ ky ] + delta_y*mac.psi_mac[ ky+1 ];

      Uc[id] =  (1.0f-delta_y)*mac.U_mac[ ky ] + delta_y*mac.U_mac[ ky+1 ];

     //   psi[id]=0.0;
      phi[id]=tanhf(psi[id]/params.sqrt2);
    }
   }    
    //std::cout<<"y= ";
    //for(int i=0+length_y; i<2*length_y; i++){
    //    std::cout<<phi[i]<<" ";
    //}
    //std::cout<<std::endl;

    setup(mac, params, length_x, length_y, x, y, phi, psi, Uc);

    //std::cout<<"y= ";
    //for(int i=0+length_y; i<2*length_y; i++){
    //    std::cout<<Uc[i]<<" ";
    //}
    //std::cout<<std::endl;
    // step 3 (time marching): call the kernels Mt times
    //ofstream out( "data.csv" );
   // out.precision(5);
    //out << phi << "\n";
   // copy( phi, phi + length, ostream_iterator<float>( out, "\n" ) );

    float cinf_cl0 =  1.0f+ (1.0f-params.k)*params.U0;
    for(int id=0; id<length; id++){

      Uc[id] = ( 1.0f+ (1.0f-params.k)*Uc[id] )*( params.k*(1.0f+phi[id])/2.0f + (1.0f-phi[id])/2.0f ) / cinf_cl0;
    }

    for(int id=0; id<length_y; id++){
      y[id] = y[id] - ztip0;
    }


    string out_format = "line_nx"+to_string(params.nx)+"_ny"+to_string(params.ny)+"_seed"+to_string(params.seed_val);
    string out_file = out_format+".h5";
    out_file = "/scratch/07428/ygqin/Aeolus/Fast_code/" + out_direc + "/" +out_file;

    // claim file and dataset handles
    hid_t  h5_file, phi_o, U_o, dataspace, xcoor, ycoor, dataspacex, dataspacey;
    hsize_t dimsf[2], dimx[1], dimy[1], dimf[1];
    herr_t  status;
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

    // write the coordinates, temperature field and the true solution to the hdf5 file
    status = H5Dwrite(phi_o, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, phi);
    status = H5Dwrite(U_o, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, Uc);
    status = H5Dwrite(xcoor, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
    status = H5Dwrite(ycoor, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, y);

    // close all the hdf handles
    H5Sclose(dataspace);H5Sclose(dataspacex);
    H5Dclose(phi_o);
    H5Dclose(U_o);
    H5Dclose(xcoor);
    H5Dclose(ycoor);H5Sclose(dataspacey);
    H5Fclose(h5_file);

    delete[] phi;
    delete[] Uc;
    delete[] psi;
    return 0;
}

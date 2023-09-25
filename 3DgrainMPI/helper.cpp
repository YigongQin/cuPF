#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "helper.h"

void print2d(float* array, int fnx, int fny){

   int length = fnx*fny;
   float* cpu_array = new float[fnx*fny];

   cudaMemcpy(cpu_array, array, length * sizeof(float),cudaMemcpyDeviceToHost);

   for (int i=0; i<length; i++){
       if (i%fnx==0) printf("\n");
       printf("%4.2f ",cpu_array[i]);
   }

}

void printCudaInfo(int rank, int i)
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);


        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("rank %d, Device %d: %s\n", rank, i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);

    printf("---------------------------------------------------------\n"); 
}


void getParam(std::string lineText, std::string key, float& param){
    std::stringstream iss(lineText);
    std::string word;
    while (iss >> word){
        //cout << word << endl;
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


float interp3Dtemperature(float* u_3d, float x, float y, float z, float Dx, int Nx, int Ny, int Nz)
{

  int kx = (int) (x/Dx);
  int ky = (int) (y/Dx);
  int kz = (int) (z/Dx);

  float delta_x = x/Dx - kx;
  float delta_y = y/Dx - ky;
  float delta_z = z/Dx - ky;

  if (kx==Nx-1) 
  {
    kx = Nx-2; 
    delta_x = 1.0f;
  }
  if (ky==Ny-1) 
  {
    ky = Ny-2; 
    delta_y =1.0f;
  }
  if (kz==Nz-1) 
  {
    kz = Nz-2; 
    delta_z =1.0f;
  }

  int offset =  kx + ky*Nx + kz*Nx*Ny;
  int offset_n =  kx + ky*Nx + (kz+1)*Nx*Ny;
  return ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset ] + (1.0f-delta_x)*delta_y*u_3d[ offset+Nx ] 
           +delta_x*(1.0f-delta_y)*u_3d[ offset+1 ] +   delta_x*delta_y*u_3d[ offset+Nx+1 ] )*(1.0f-delta_z) + 
         ( (1.0f-delta_x)*(1.0f-delta_y)*u_3d[ offset_n ] + (1.0f-delta_x)*delta_y*u_3d[ offset_n+Nx ] 
           +delta_x*(1.0f-delta_y)*u_3d[ offset_n+1 ] +   delta_x*delta_y*u_3d[ offset_n+Nx+1 ] )*delta_z;
}

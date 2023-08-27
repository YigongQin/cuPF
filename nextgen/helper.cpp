#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>

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

template <typename T>
std::string to_stringp(const T a_value, int n )
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}
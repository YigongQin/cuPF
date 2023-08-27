#pragma once

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

void print2d(float* array, int fnx, int fny);
void printCudaInfo(int rank, int i);
void getParam(std::string lineText, std::string key, float& param);
void read_input(std::string input, float* target);
void read_input(std::string input, int* target);

__global__ void set_minus1(float* u, int size);
__global__ void ave_x(float* phi, float* meanx, int fnx, int fny, int fnz, int NUM_PF);
void tip_mvf(int *cur_tip, float* phi, float* meanx, float* meanx_host, int fnx, int fny, int fnz, int NUM_PF);


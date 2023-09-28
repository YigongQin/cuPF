//////////////////////
// ThermalInputData.h
// COPYRIGHT Yigong Qin, ALL RIGHTS RESERVED
/////////////////////
#pragma once
#include <string>

class ThermalInputData
{
public:
    void readThermalInputData();

    std::string folder;
    int Nx, Ny, Nz, Nt;
    float* X_mac; 
    float* Y_mac; 
    float* Z_mac;
    float* t_mac;
    int* alpha_mac;
    float* psi_mac;
    float* T_3D;
    float* theta_arr;
    float* cost;
    float* sint;
};

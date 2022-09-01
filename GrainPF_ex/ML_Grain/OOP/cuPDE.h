#ifndef __CUPDE_H__
#define __CUPDE_H__

#include "params.h"

class PDE{

public:
	float* x;
	float* y;
	float* z;
 	float* x_device;
    float* y_device;
    float* z_device;
	float* t;
	
	virtual ~PDE(){};
	virtual void cpuSetup(params_MPI pM, GlobalConstants params);
	virtual void initField(Mac_input mac);
	virtual void cudaSetup(params_MPI pM); // setup cuda for every GPU
	//virtual void evolve() = 0;
	virtual void output() = 0;

};


#endif
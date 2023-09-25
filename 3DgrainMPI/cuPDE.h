#pragma once
#include "MPIsetting.h"

class PDE
{
public:
	virtual ~PDE(){};
	virtual void cpuSetup(MPIsetting* mpiManager) = 0;
	virtual void initField() = 0;
	virtual void cudaSetup() = 0; // setup cuda for every GPU
	virtual void evolve() = 0;

protected:
	float* x;
	float* y;
	float* z;
	float* x_device;
	float* y_device;
	float* z_device;
	float* t;
};

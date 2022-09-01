#ifndef __CUPDE_H__
#define __CUPDE_H__

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
	virtual void cpuSetup() = 0;
	virtual void initField() = 0;
	virtual void cudaSetup() = 0; // implement later
	virtual void evolve() = 0;
	virtual void output() = 0;

};


#endif
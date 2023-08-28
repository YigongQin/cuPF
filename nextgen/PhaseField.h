#pragma once

#include "cuPDE.h"
#include "params.h"
#include "QOI.h"
#include "thermalInputData.h"

class PhaseField: public PDE 
{
public:
	PhaseField(){};
	virtual ~PhaseField();
	void parseInputParams(const std::string fileName);
	void cpuSetup(MPIsetting& mpiManager);
	void initField();
	virtual void cudaSetup(const MPIsetting& mpiManager); // setup cuda for every GPU
	virtual void evolve(); // evolve the field with input
	void output(const MPIsetting& mpiManager, const std::string outputFolder, bool save3DField); 

	// grid size
	int fnx, fny, fnz, fnz_f, NUM_PF, length, full_length;

	// host pointers

	float* z_full;
	float* phi;
	float* psi;
	float* Uc;
	int* alpha;
	int* alpha_i_full;

	// device pointers
	float* phi_old;
	float* phi_new;
	float* PFs_old;
	float* PFs_new;
	int* alpha_m;
	int* d_alpha_full;
	int* argmax;
	float* z_device2;

	thermalInputData mac;
	thermalInputData Mgpu;
	GlobalConstants params;
	QOI* qois;

};
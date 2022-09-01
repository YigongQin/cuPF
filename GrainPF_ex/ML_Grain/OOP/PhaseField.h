#ifndef __PHASEFIELD_H__
#define __PHASEFIELD_H__

#include "cuPDE.h"
#include "params.h"

class PhaseField: public cuPDE {

public:
	// define the scale/resolution of the problem first
	int fnx, fny, fnz, fnz_f, NUM_PF, length, full_length;

	float* z_device2;
	float* phi;
	float* psi;
	float* U;
	int* alpha;
	int* alpha_i_full;

	// device pointers
	float* phi_old;
	float* phi_new;
	float* PFs_old;
	float* PFs_new;
	float* alpha_m;
	int* d_alpha_full;
	int* argmax;


	PhaseField();
	virtual ~PhaseField();
	void cpuSetup(params_MPI pM, GlobalParams params);
	void initField(Mac_input mac);
	void cudaSetup(params_MPI pM); // setup cuda for every GPU
	void evolve(GlobalParams params, Mac_input mac, float* tip_y, float* frac, int* aseq, int* extra_area, int* tip_final, int* total_area, int* cross_sec); // evolve the field with input
	void output();

}



#endif
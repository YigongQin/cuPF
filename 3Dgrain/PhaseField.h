#ifndef __PHASEFIELD_H__
#define __PHASEFIELD_H__

#include "cuPDE.h"
#include "params.h"
#include "QOI.h"
#include <string>
using namespace std;

class PhaseField: public PDE {

public:
	Mac_input mac;
	Mac_input Mgpu;
	GlobalConstants params;
	QOI* q;
	// define the scale/resolution of the problem first
	int fnx, fny, fnz, fnz_f, NUM_PF, length, full_length;

	float* z_device2;

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


	PhaseField(){};
	virtual ~PhaseField();
	void parseInputParams(char* fileName);
	void cpuSetup(params_MPI &pM);
	void initField();
	virtual void cudaSetup(params_MPI pM); // setup cuda for every GPU
	virtual void evolve(); // evolve the field with input
	void output(params_MPI pM);

};

void calc_qois(int* cur_tip, int* alpha, int fnx, int fny, int fnz, int kt, int num_grains, \
  float* tip_z, int* cross_sec, float* frac, float* z, int* ntip, int* extra_area, int* tip_final, int* total_area, int* loss_area, int move_count, int all_time);

#endif
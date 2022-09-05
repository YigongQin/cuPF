#ifndef __APTPHASEFIELD_H__
#define __APTPHASEFIELD_H__

#include "cuPDE.h"
#include "params.h"
#include "QOI.h"
#include "PhaseField.h"
#include <string>
using namespace std;

class APTPhaseField: public PhaseField {

public:
	int* active_args;

	APTPhaseField(){};
	virtual ~APTPhaseField();
	void cudaSetup(params_MPI pM); // setup cuda for every GPU
	void evolve(); // evolve the field with input


};


#endif
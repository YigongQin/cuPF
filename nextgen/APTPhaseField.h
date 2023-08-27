//////////////////////
// APTPhaseField.h
// COPYRIGHT Yigong Qin, ALL RIGHTS RESERVED
/////////////////////

#pragma once

#include "PhaseField.h"


class APTPhaseField: public PhaseField 
{
public:
	APTPhaseField(){};
	virtual ~APTPhaseField();
	void cudaSetup(const MPIsetting& mpiManager); // setup cuda for every GPU
	void evolve(); // evolve the field with input

private:
	int* active_args_old;
	int* active_args_new;
	int* args_cpu;
};


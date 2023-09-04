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
	void cudaSetup(); // setup cuda for every GPU
	void evolve(); // evolve the field with input


protected:
	virtual void moveDomain(MovingDomain* movingDomainManager);
	virtual void getLineQoIs(MovingDomain* movingDomainManager);
	
private:
	int* active_args_old;
	int* active_args_new;
	int* args_cpu;
};


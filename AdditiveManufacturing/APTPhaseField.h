//////////////////////
// APTPhaseField.h
// COPYRIGHT Yigong Qin, ALL RIGHTS RESERVED
/////////////////////

#pragma once

#include "PhaseField.h"
#include "MovingDomain.h"

class APTPhaseField: public PhaseField 
{
public:
	APTPhaseField(){};
	virtual ~APTPhaseField();
	void cudaSetup(); // setup cuda for every GPU
	void evolve(); // evolve the field with input


protected:
	virtual void initPhaseFieldFromliquid(){};
	virtual void moveDomain(MovingDomain* movingDomainManager);
	virtual void getLineQoIs(MovingDomain* movingDomainManager);
	void setBC(bool useLineConfig, float* ph, int* active_args);

private:
	int* active_args_old;
	int* active_args_new;
	int* args_cpu;
};


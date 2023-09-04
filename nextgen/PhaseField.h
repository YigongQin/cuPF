#pragma once

#include "cuPDE.h"
#include "params.h"
#include "QOI.h"
#include "ThermalInputData.h"
#include "DesignSettingData.h"

class MovingDomain;

class PhaseField: public PDE 
{
public:
	PhaseField(){};
	virtual ~PhaseField();
	void parseInputParams(const std::string fileName);
	void cpuSetup(MPIsetting* mpiManager);
	void initField();
	virtual void cudaSetup(); // setup cuda for every GPU
	virtual void evolve(); // evolve the field with input
    void output(const std::string outputFolder, bool save3DField); 
	inline void SetMPIManager(MPIsetting* mpiManager);
	inline MPIsetting* GetMPIManager() const;
	inline void SetDesignSetting(const DesignSettingData* designSetting);
	inline const DesignSettingData* GetSetDesignSetting() const;
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

	ThermalInputData mac;
	ThermalInputData Mgpu;
	GlobalConstants params;
	QOI* qois;

	MPIsetting* mMPIManager;
	const DesignSettingData* mDesignSetting;

protected:
	virtual void moveDomain(MovingDomain* movingDomainManager);
	virtual void getLineQoIs(MovingDomain* movingDomainManager);
};

inline void PhaseField::SetMPIManager(MPIsetting* mpiManager)
{
	mMPIManager = mpiManager;
}

inline MPIsetting* PhaseField::GetMPIManager() const
{
	return mMPIManager;
}

inline void PhaseField::SetDesignSetting(const DesignSettingData* designSetting)
{
	mDesignSetting = designSetting;
}


inline const DesignSettingData* PhaseField::GetSetDesignSetting() const
{
	return mDesignSetting;
}


class MovingDomain
{
public:
	MovingDomain();
	void allocateMovingDomain(int numGrains, int MovingDirectoinSize);

	int move_count;
    int cur_tip;
    int tip_front;
    int tip_thres;
    int samples, lowsl;
	float* meanx; 
	float* meanx_host;
	int* loss_area; 
	int* d_loss_area;
};

MovingDomain::MovingDomain()
: move_count(0), cur_tip(1), tip_front(1),  samples(0), lowsl(1)
{
}
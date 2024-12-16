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
    void OutputQoIs();
	void OutputField(int currentStep);
	inline void SetMPIManager(MPIsetting* mpiManager);
	inline MPIsetting* GetMPIManager() const;
	inline void SetDesignSetting(const DesignSettingData* designSetting);
	inline const DesignSettingData* GetSetDesignSetting() const;
	// grid size
	int fnx, fny, fnz, fnz_f, NUM_PF, length, full_length;
	int blocksize_1d, blocksize_2d, num_block_2d, num_block_PF,num_block_PF1d, max_area;
	int cnx, cny, cnz, num_block_c;
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
	int* nucleationStatus;

	MPIsetting* mMPIManager;
	const DesignSettingData* mDesignSetting;

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


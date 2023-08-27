//////////////////////
// MPIsetting.h
// COPYRIGHT Yigong Qin, ALL RIGHTS RESERVED
/////////////////////

#pragma once

class MPIsetting
{
public:
	void printMPIsetting() const;
	virtual void oneDimensionPartition(){};
	virtual void twoDimensionPartition();
	virtual void ThreeDimensionPartition(){};

    int rank;
    int processorIDX, processorIDY, processorIDZ;
    int numProcessor, numProcessorX, numProcessorY, numProcessorZ;
    int nxLocal, nyLocal, nzLocal, nzLocalAll;
};


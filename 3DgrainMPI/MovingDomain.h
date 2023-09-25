#pragma once

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
	float* meanx_device; 
	float* meanx_host;
	int* loss_area_device; 
	int* loss_area_host;
};


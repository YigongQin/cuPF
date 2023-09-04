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

MovingDomain::MovingDomain()
: move_count(0), cur_tip(1), tip_front(1),  samples(0), lowsl(1)
{
}
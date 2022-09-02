#ifndef __QOI_H__
#define __QOI_H__
#include "params.h"
class QOI{

public:
	int num_case;
	int valid_run;
	float *tip_y, *frac, *angles;
	int *alpha, *extra_area, *total_area, *tip_final, *cross_sec;

	QOI(){};
	virtual ~QOI();
	void initQoI();
	

};

void QOI::initQoI(GlobalConstants params){
    tip_y = new float[num_case*(params.nts+1)];
    frac = new float[num_case*(params.nts+1)*params.num_theta];
    angles = new float[num_case*(2*params.NUM_PF+1)];

    cross_sec = new int[num_case*(params.nts+1)*params.fnx*params.fny];
    alpha = new int[valid_run*params.full_length];
    extra_area = new int[num_case*(params.nts+1)*params.num_theta];
    total_area  = new int[num_case*(params.nts+1)*params.num_theta];
    tip_final   = new int[num_case*(params.nts+1)*params.num_theta];
}




#endif
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
	virtual ~QOI(){};
	void initQoI(GlobalConstants params);
	

};



#endif
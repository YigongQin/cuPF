//////////////////////
// QOI.h
// COPYRIGHT Yigong Qin, ALL RIGHTS RESERVED
/////////////////////

#pragma once

#include "params.h"
#include <vector>
#include <map>

class QOI
{
public:
	QOI(GlobalConstants params);
	virtual ~QOI(){};
	void searchJunctionsOnImage(const GlobalConstants& params, const int* alpha);
	void calculateLineQoIs(const GlobalConstants& params, int& cur_tip, const int* alpha, int kt, 
                        const float* z, const int* loss_area, int move_count);
	void sampleHeights(int& cur_tip, const int* alpha, int fnx, int fny, int fnz);
	std::map<std::string, std::vector<int> >   mQoIVectorIntData;
	std::map<std::string, std::vector<float> > mQoIVectorFloatData;
	std::map<std::string, int > mQoIVectorSize;

private:
	int mNumNodeFeatures = 8;
};


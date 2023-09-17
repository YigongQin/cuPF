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
    virtual void searchJunctionsOnImage(const GlobalConstants& params, const int* alpha) = 0;
	void calculateLineQoIs(const GlobalConstants& params, int& cur_tip, const int* alpha, int kt, 
                        const float* z, const int* loss_area, int move_count);
	void calculateQoIs(const GlobalConstants& params, const int* alpha, int kt);
	void sampleHeights(int& cur_tip, const int* alpha, int fnx, int fny, int fnz);
	std::map<std::string, std::vector<int> >   mQoIVectorIntData;
	std::map<std::string, std::vector<float> > mQoIVectorFloatData;
	std::map<std::string, int > mQoIVectorSize;
};

class QOILine : public QOI
{
public:
	QOILine(GlobalConstants params);
	virtual ~QOILine(){};
	void searchJunctionsOnImage(const GlobalConstants& params, const int* alpha) override;
private:
	int mNumNodeFeatures = 8;
};

class QOI3D : public QOI
{
public:
	QOI3D(GlobalConstants params);
	virtual ~QOI3D(){};
	void searchJunctionsOnImage(const GlobalConstants& params, const int* alpha) override;

private:
	int mNumNodeFeatures = 9;
};
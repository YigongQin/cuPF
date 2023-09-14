//////////////////////
// DesignSettingData.h
// COPYRIGHT Yigong Qin, ALL RIGHTS RESERVED
/////////////////////
#pragma once
#include <string>

class DesignSettingData
{
public:
    DesignSettingData();
    void getOptions(int argc, char** argv);

    bool useMPI;
    bool useLineConfig;
    bool includeNucleation;
    bool pureNucleation;
    bool useAPT;
    bool save3DField;

    int mpiDim;
    int seedValue;
    int bcX, bcY, bcZ;

    std::string inputFile;
    std::string thermalInputFolder;
    std::string outputFolder;
};
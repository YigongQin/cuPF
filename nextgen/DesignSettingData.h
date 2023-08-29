//////////////////////
// DesignSettingData.h
// COPYRIGHT Yigong Qin, ALL RIGHTS RESERVED
/////////////////////
#pragma once
#include <string>

class DesignSettingData
{
public:

    void getOptions(int argc, char** argv);

    bool useMPI;
    bool useMovingFrame;
    bool includeNucleation;
    bool useAPT;
    bool save3DField;

    int  mpiDim;
    int  seedValue;

    std::string inputFile;
    std::string thermalInputFolder;
    std::string outputFolder;
};
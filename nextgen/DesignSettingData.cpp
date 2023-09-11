#include "DesignSettingData.h"
#include <getopt.h>
#include <string>
#include <iostream>
#include <functional>
using namespace std;

DesignSettingData::DesignSettingData()
: useMPI(false), useLineConfig(false), includeNucleation(false), useAPT(true), save3DField(false), mpiDim(1), seedValue(0)
{
}

void DesignSettingData::getOptions(int argc, char** argv)
{
    inputFile = argv[1]; 
    thermalInputFolder = "forcing/case";
    // this output folder should be specified differently for each system
    outputFolder = "/scratch/07428/ygqin/graph/"; 

    static struct option long_options[] = 
    {
        {"help",     0, 0,  '?'},
        {"macfile",  1, 0,  'f'},
        {"useAPT",   1, 0,  'a'},
        {"seed",     1, 0,  's'},
        {"output",   1, 0,  'o'},
        {"mpiDim",   1, 0,  'm'},
        {"savebulk", 1, 0,  'b'},
        {"lineConfig",     1, 0,  'l'},
        {"includeNuclean", 1, 0,  'n'},
        {0 ,0, 0, 0}
    };

    int opt;    


    while ((opt = getopt_long(argc, argv, "b:f:o:a:s:m:l:n?", long_options, NULL)) != EOF) 
    {
        switch (opt) 
        {
            case 'f':
                thermalInputFolder = optarg;
                break;
            case 'a':
                useAPT = false;
                break;
            case 's':
                seedValue = atoi(optarg);
                break;
            case 'o':
                outputFolder = outputFolder + optarg;
                break;
            case 'm':
                mpiDim = atoi(optarg);
                break;
            case 'b':
                save3DField = true;    
                cout << "save entire 3D data" << endl;
                break;
            case 'l':
                useLineConfig = true;
                break;
            case 'n':
                includeNucleation = true;
                break;            
        }
    }
}
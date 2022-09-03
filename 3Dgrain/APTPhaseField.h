#ifndef __APTPHASEFIELD_H__
#define __APTPHASEFIELD_H__

#include "PhaseField.h"
#include "params.h"
#include "QOI.h"
#include <string>
using namespace std;

class APTPhaseField: public PhaseField {

public:
	int* aptArgs;

};



#endif
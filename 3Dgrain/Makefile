# set the of for different platforms
CXX=icpc 
CUDA_PATH=$(TACC_CUDA_DIR)
HDF5_INC=$(TACC_HDF5_INC) 
HDF5_LIB=$(TACC_HDF5_LIB)

###########################################################


EXECUTABLE := phase_field
CU_FILES   := PhaseField.cu APTPhaseField.cu 
CU_DEPS    :=
CC_FILES   := main.cpp helper.cpp PhaseFieldCpu.cpp

all: $(EXECUTABLE)

LOGS	   := logs

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
INC_FLAGS=-I../common -I$(HDF5_INC) 
CXXFLAGS=-O3 -Wall $(INC_FLAGS) -std=c++14
HDF5_FLAGS=-Wl,-rpath,$(HDF5_LIB) -L$(HDF5_LIB) -lhdf5 -lz

################################################################################
# When compiling with NVCC, the arch flag (-arch) specifies the name of 
# the NVIDIA GPU architecture that the CUDA files will be compiled for.
# Gencodes (-gencode) allows for more PTX generations, and can be repeated 
# many times for different architectures.
# Pascal (CUDA 8 and later)
# 	SM60 or SM_60, compute_60 – Quadro GP100, Tesla P100, DGX-1 (Generic Pascal)
# 	SM61 or SM_61, compute_61 – GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# Volta (CUDA 9 and later)
# 	SM70 or SM_70, compute_70 – DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100
# 	SM72 or SM_72, compute_72 – Jetson AGX Xavier
#
# CUDA code generation flags
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
GENCODE_FLAGS   := $(GENCODE_SM70) # For Pascal architecture
################################################################################

NVCC_LDFLAGS=-L$(CUDA_PATH)/lib64/ -lcudart -Wl,-rpath=$(CUDA_PATH)/lib64
NVCC=$(CUDA_PATH)/bin/nvcc -ccbin=$(CXX) 
NVCC_FLAGS=-O3 -m64 $(GENCODE_FLAGS) $(INC_FLAGS) -std=c++14


OBJS=$(OBJDIR)/main.o $(OBJDIR)/helper.o $(OBJDIR)/PhaseFieldCpu.o
OBJS += $(OBJDIR)/PhaseField.o $(OBJDIR)/APTPhaseField.o 

CXXFLAGS += -I$(CUDA_PATH)/include/ $(HDF5_FLAGS) -L$(CUDA_PATH)/lib64/ -lcudart 


.PHONY: dirs clean

default: $(EXECUTABLE) 

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)



$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(NVCC_LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCC_FLAGS) -c -o $@

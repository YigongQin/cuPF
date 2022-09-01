EXECUTABLE := phase_field
CU_FILES   := PhaseField.cu
CU_DEPS    :=
CC_FILES   := main.cpp helper.cpp PhaseFieldCpu.cpp

all: $(EXECUTABLE)

LOGS	   := logs

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
CUDA_PATH=$(TACC_CUDA_DIR)
OBJDIR=objs
INC_FLAGS=-I../common -I$(TACC_HDF5_INC) 
MPI_INC=-I/opt/intel/impi/2019.9.304/intel64/include/
CXX=icpc 
CXXFLAGS=-O3 -Wall $(INC_FLAGS) $(MPI_INC) -std=c++11
HDF5_FLAGS=-Wl,-rpath,$(TACC_HDF5_LIB) -L$(TACC_HDF5_LIB) -lhdf5 -lz
MPICXX=/opt/intel/impi/2019.9.304/intel64/bin/mpicxx
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
NVCC=$(CUDA_PATH)/bin/nvcc -ccbin=$(MPICXX) 
NVCC_FLAGS=-O3 -m64 $(GENCODE_FLAGS) $(INC_FLAGS)

#CXXFLAGS += -I$(CUDA_PATH)/include/ -L$(CUDA_PATH)/lib64/ -lcudart

OBJS=$(OBJDIR)/main.o $(OBJDIR)/helper.o 
OBJS += $(OBJDIR)/PhaseField.o 

CXXFLAGS += -I$(CUDA_PATH)/include/ $(HDF5_FLAGS) -L$(CUDA_PATH)/lib64/ -lcudart 


.PHONY: dirs clean

default: $(EXECUTABLE) 

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)



$(EXECUTABLE): dirs $(OBJS)
		$(MPICXX) $(CXXFLAGS) -o $@ $(OBJS) $(NVCC_LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(MPICXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCC_FLAGS) -c -o $@


# Naive CUDA search
NVCC_DEBIAN=/usr/bin/nvcc
NVCC75=/usr/local/cuda-7.5/bin/nvcc
NVCC70=/usr/local/cuda-7.0/bin/nvcc
NVCC65=/usr/local/cuda-6.5/bin/nvcc
NVCC60=/usr/local/cuda-6.0/bin/nvcc


NVCCFLAGS_5    = -gencode arch=compute_53,code=sm_53                             
NVCCFLAGS_5    += -gencode=arch=compute_53,code=compute_53                       
NVCCFLAGS_5    += -gencode arch=compute_52,code=sm_52                             
NVCCFLAGS_5    +=  -gencode=arch=compute_52,code=compute_52                       
NVCCFLAGS_5    += -gencode arch=compute_50,code=sm_50                             
NVCCFLAGS_5    +=  -gencode=arch=compute_50,code=compute_50

NVCCFLAGS    = -gencode arch=compute_35,code=sm_35
NVCCFLAGS    += -gencode arch=compute_35,code=sm_35
NVCCFLAGS    += --compiler-options=-Wall,-Wno-unused-function -I$(CURDIR) -O3 -std=c++11 -ccbin=/usr/bin/g++-4.9

ifneq ("$(wildcard $(NVCC75))","")
	NVCC = $(NVCC75)
	NVCCFLAGS += $(NVCCFLAGS_5)
else ifneq ("$(wildcard $(NVCC70))","")
	NVCC = $(NVCC70)
	NVCCFLAGS += $(NVCCFLAGS_5)
else ifneq ("$(wildcard $(NVCC65))","")
	NVCC = $(NVCC65)
else ifneq ("$(wildcard $(NVCC_DEBIAN))","")
	NVCC = $(NVCC_DEBIAN)
else
	NVCC = $(NVCC60)
endif

# Use ctags if available
CTAGS=ctags

NVCCLIBSFLAGS = -dc 

TOOLS_SRC=$(wildcard tools/*.cu)
COMPRESSION_SRC=$(wildcard compression/*.cu)

COMPRESSION_LIB_OBJ_BASE=
COMPRESSION_LIB_OBJ_CPU =
COMPRESSION_LIB_OBJ_GPU = $(TOOLS_SRC:.cu=.o) $(COMPRESSION_SRC:.cu=.o)

GPU_LIBS =  $(COMPRESSION_LIB_OBJ_GPU)
CPU_LIBS =  $(COMPRESSION_LIB_OBJ_CPU)
ALL_LIBS =  $(COMPRESSION_LIB_OBJ_BASE) 


TESTS_SRC=$(wildcard tests/test*.cu)
TESTS_OBJ = $(TESTS_SRC:.cu=.o)
TESTS_RUNER = tests/run_tests.out

#PROGS = multi_gpu_transfer.out compression_tests.out test.out

all:$(PROGS) $(TESTS_RUNER) 

debug: NVCCFLAGS += -g -G 
debug: all

verbose: NVCCFLAGS += -Xptxas="-v"
verbose: all

clean:
	rm -f $(CPU_LIBS) $(GPU_LIBS) $(ALL_LIBS) $(PROGS) $(TESTS_OBJ) $(TESTS_RUNER) *.o *.pyc tags gpu_compression_lib.tar.bz2

$(PROGS): %.out: %.o $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS)
	$(NVCC) $(NVCCFLAGS)  $< $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS) -o $@

$(TESTS_RUNER): %.out: %.o $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS) $(TESTS_OBJ)
	$(NVCC) $(NVCCFLAGS)  $< $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS) $(TESTS_OBJ) -o $@

.SUFFIXES: .cu .cuh .out .o

$(TESTS_OBJ): %.o : %.cu %.cuh tests/test_base.cuh

.cu.o: %.cu %.h %.cuh 
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBSFLAGS) $< -o $@

ctags:
	@$(CTAGS) --langmap=c++:+.cu --append=no *.cu*  compression/*.cu* tools/*.cu* tests/*.cu* 2>&1 /dev/null

fixcuda:
	sudo nvidia-smi -pm 1
export:
	git archive HEAD --prefix gpu_compression_lib/ | bzip2 > gpu_compression_lib.tar.bz2

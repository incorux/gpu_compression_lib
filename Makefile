NVCC=nvcc

NVCCLIBSFLAGS = -dc 
#NVCCFLAGS    = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
#NVCCFLAGS    = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35
NVCCFLAGS    = -gencode arch=compute_35,code=sm_35
NVCCFLAGS    += --compiler-options=-Wall,-Wno-unused-function -I$(CURDIR) -O3

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

PROGS = multi_gpu_transfer.out compression_tests.out test.out

all:$(PROGS) $(TESTS_RUNER)

debug: NVCCFLAGS += -g -G 
debug: ctags $(PROGS) 

verbose: NVCCFLAGS += -Xptxas="-v"
verbose: ctags $(PROGS) 

clean:
	rm -f $(CPU_LIBS) $(GPU_LIBS) $(ALL_LIBS) $(PROGS) *.o *.pyc tags gpu_compression_lib.tar.bz2

$(PROGS): %.out: %.o $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS)
	$(NVCC) $(NVCCFLAGS)  $< $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS) -o $@

$(TESTS_RUNER): %.out: %.o $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS) $(TESTS_OBJ)
	$(NVCC) $(NVCCFLAGS)  $< $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS) $(TESTS_OBJ) -o $@

.SUFFIXES: .cu .out .o

.cu.o: %.cu %.h
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBSFLAGS) $< -o $@

ctags:
	@ctags --langmap=c++:+.cu --append=no *.cu*  compression/*.cu* tools/*.cu* tests/*.cu* 2>&1 /dev/null

fixcuda:
	sudo nvidia-smi -pm 1
export:
	git archive HEAD --prefix gpu_compression_lib/ | bzip2 > gpu_compression_lib.tar.bz2

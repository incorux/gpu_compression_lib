NVCC=nvcc

NVCCLIBSFLAGS = -dc 
NVCCFLAGS    = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
NVCCFLAGS    += --compiler-options=-Wall,-Wno-unused-function -I$(CURDIR) -Ithrust -O3


COMPRESSION_LIB_OBJ_BASE=
COMPRESSION_LIB_OBJ_CPU =
COMPRESSION_LIB_OBJ_GPU = compression/avar_gpu.o compression/tools.o

GPU_LIBS =  $(COMPRESSION_LIB_OBJ_GPU)
CPU_LIBS =  $(COMPRESSION_LIB_OBJ_CPU)
ALL_LIBS =  $(COMPRESSION_LIB_OBJ_BASE) 

PROGS = multi_gpu_transfer.out compression_tests.out 

all:$(PROGS) 

debug: NVCCFLAGS += -g -G -DTHRUST_DEBUG
debug: ctags $(PROGS) 

clean:
	rm -f $(CPU_LIBS) $(GPU_LIBS) $(ALL_LIBS) $(PROGS) *.o *.pyc tags gpu_compression_lib.tar.bz2

$(PROGS): %.out: %.o $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS)
	$(NVCC) $(NVCCFLAGS)  $< $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS) -o $@

.SUFFIXES: .cu .out .o

.cu.o: %.cu %.h
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBSFLAGS) $< -o $@

ctags:
	@ctags --langmap=c++:+.cu --append=no *.cu compression/*.{cu,cuh} 2>&1 /dev/null

fixcuda:
	sudo nvidia-smi -pm 1
export:
	git archive master --prefix gpu_compression_lib/ | bzip2 > gpu_compression_lib.tar.bz2

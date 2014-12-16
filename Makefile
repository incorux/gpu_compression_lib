NVCC=nvcc

NVCCLIBSFLAGS = -dc 
NVCCFLAGS     = --compiler-options=-Wall,-Wno-unused-function -arch sm_20 -I$(CURDIR) -Ithrust -O3
#NVCCFLAGS     = --compiler-options=-Wall,-Wno-unused-function -arch sm_20 -I$(CURDIR) -Ithrust -g -G 


COMPRESSION_LIB_OBJ_BASE=
COMPRESSION_LIB_OBJ_CPU =
COMPRESSION_LIB_OBJ_GPU = compression/avar_gpu.o compression/tools.o

GPU_LIBS =  $(COMPRESSION_LIB_OBJ_GPU)
CPU_LIBS =  $(COMPRESSION_LIB_OBJ_CPU)
ALL_LIBS =  $(COMPRESSION_LIB_OBJ_BASE) 

PROGS = multi_gpu_transfer.out compression_tests.out 

all:$(PROGS) 

debug: NVCCFLAGS += -g -DTHRUST_DEBUG
debug: ctags $(PROGS) 

clean:
	rm $(CPU_LIBS) $(GPU_LIBS) $(ALL_LIBS) $(PROGS) *.o *.pyc tags

$(PROGS): %.out: %.o $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS)
	$(NVCC) $(NVCCFLAGS)  $< $(GPU_LIBS) $(CPU_LIBS) $(ALL_LIBS) -o $@

.SUFFIXES: .cu .out .o

.cu.o: %.cu %.h
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBSFLAGS) $< -o $@

ctags:
	@ctags --langmap=c++:+.cu --append=no **/*.{cu,cuh}  2>&1 /dev/null

fixcuda:
	sudo nvidia-smi -pm 1

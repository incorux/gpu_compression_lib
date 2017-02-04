# Naive and dirty CUDA setup
# This manages setup diffrences on various of our test machines as on some of them we have only limited access

NVCC_DEBIAN=/usr/bin/nvcc
NVCC80=/usr/local/cuda-8.0/bin/nvcc
NVCC75=/usr/local/cuda-7.5/bin/nvcc
NVCC70=/usr/local/cuda-7.0/bin/nvcc
NVCC65=/usr/local/cuda-6.5/bin/nvcc
NVCC60=/usr/local/cuda-6.0/bin/nvcc

NVCCFLAGS_5    =  -gencode arch=compute_53,code=sm_53                             
NVCCFLAGS_5    += -gencode=arch=compute_53,code=compute_53                       
NVCCFLAGS_5    += -gencode arch=compute_52,code=sm_52                             
NVCCFLAGS_5    += -gencode=arch=compute_52,code=compute_52                       
NVCCFLAGS_5    += -gencode arch=compute_50,code=sm_50                             
NVCCFLAGS_5    += -gencode=arch=compute_50,code=compute_50

NVCCFLAGS    = -gencode arch=compute_35,code=sm_35
NVCCFLAGS    += -gencode arch=compute_35,code=sm_35
NVCCFLAGS    += -D_GLIBCXX_USE_CXX11_ABI=0 --compiler-options=-Wall,-Wno-unused-function -I$(CURDIR)
# NVCCFLAGS    += --linker-options=--gc-sections,--print-gc-sections 
NVCCFLAGS 	 += -std=c++11 -D_FORCE_INLINES  -O3  

ifneq ("$(wildcard $(NVCC80))","")
	NVCC = $(NVCC80)
	NVCCFLAGS += $(NVCCFLAGS_5)
else ifneq ("$(wildcard $(NVCC75))","")
	NVCC = $(NVCC75)
	NVCCFLAGS += $(NVCCFLAGS_5)
else ifneq ("$(wildcard $(NVCC70))","")
	NVCC = $(NVCC70)
	NVCCFLAGS += $(NVCCFLAGS_5)
else ifneq ("$(wildcard $(NVCC_DEBIAN))","")
	NVCC = $(NVCC_DEBIAN)
	NVCCFLAGS += $(NVCCFLAGS_5)
else ifneq ("$(wildcard $(NVCC65))","")
	NVCC = $(NVCC65)
else
	NVCC = $(NVCC60)
endif

# FIX for default gcc for various machines
ifneq ("$(wildcard /usr/bin/g++-4.9)","")
	NVCCFLAGS    += -ccbin=/usr/bin/g++-4.9  
else ifneq ("$(wildcard /home/samba/przymusp/src/gcc-4.9.4/gcc/bin/g++)","")
	NVCCFLAGS    += -ccbin=/home/samba/przymusp/src/gcc-4.9.4/gcc/bin/g++
endif


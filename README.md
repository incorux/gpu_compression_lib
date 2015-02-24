# Description:
Two variants of FL algorithm 
  * AFL - aligned version of Fixed length algorithm, it is up to 10x faster then non-aligned version of this algorithm
  * FL - nonaligned version of Fixed length algorithm

Version of algorithm is controlled as CWARP_SIZE template constants:
  * FL_ALGORITHM_MOD_FL=1 for FL version of algorithm,
  * FL_ALGORITHM_MOD_AFL=32 for AFL version of algorithm.

It is given as a template parameter in order to allow the compiler to optimize divide and modulo operations.

Supported data types: int, unsigned int, long, unsigned long
Available functions are force compiled into afl_gpu.o binary.

# Building and dependencies
Use make to build. 
Cuda 6.5 is needed to compile.

# Code:
  * compression/avar_gpu.[cu,cuh]
  * macros.h

# Examples included:
  * multi_gpu_transfer.cu - multi GPU transfer for p2p enabled devices
  * compression_tests.cu - compression/decompression test code 

# Example usage:
See compression_tests.cu

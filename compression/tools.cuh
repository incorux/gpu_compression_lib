#ifndef _TOOLS
#define _TOOLS true
#include "tools.cuh"
#include "macros.cuh"
#include <stdio.h>
#include <list>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define cudaErrorCheck()  { _cudaErrorCheck(__FILE__, __LINE__); }
inline void _cudaErrorCheck(const char *file, int line)
{
    gpuAssert( cudaPeekAtLastError(), file, line );
    gpuAssert( cudaDeviceSynchronize(), file, line );
}

inline int pow(int val, int pow)
{
    return std::pow((double)val, pow);
}

inline void big_random_block( unsigned long size, int limit_bits, int *data)
{
    unsigned int mask = NBITSTOMASK(limit_bits);
    for (unsigned long i = 0; i < size; i++)
        data[i] = rand() & mask;
}

typedef struct allocation_info
{
    void *data;
    unsigned long size;
    char device;
    bool freed;
} allocation_info;

typedef std::list<struct allocation_info> mmManager;

void mmCudaMallocHost(mmManager &manager, void **data, unsigned long size);
void mmCudaMalloc(mmManager &manager, void **data, unsigned long size);
void mmCudaFreeAll(mmManager &manager);
void mmCudaFree(mmManager &manager, void *ptr);

typedef struct timeit_info
{
    float __elapsedTime; 
    cudaEvent_t __start;
    cudaEvent_t __stop;
    char *name;
} timit_info;

typedef std::list<timeit_info *> tiManager;

void tiStart(tiManager &manager);
void tiEnd(tiManager &manager, const char * name);
void tiPreatyPrint(tiManager &manager);
void tiClear(tiManager &manager);
void tiPreatyPrintThrougput(tiManager &manager, int data_size);


#define DEBUG 1
#ifdef DEBUG
# define DPRINT(x) printf x
#else
# define DPRINT(x) do {} while (0)
#endif

int compare_arrays(int *in1, int *in2, unsigned long size);
int compare_arrays_float(float *in1, float *in2, unsigned long size);

#define TIMEIT_SETUP() tiManager __tim__;
#define TIMEIT_START() tiStart(__tim__);
#define TIMEIT_END(name) tiEnd(__tim__, name);
#define TIMEIT_PRINT() tiPreatyPrint(__tim__); tiClear(__tim__);
#define TIMEIT_PRINT_THROUGPUT(data_size) tiPreatyPrintThrougput(__tim__, data_size); tiClear(__tim__);
#endif

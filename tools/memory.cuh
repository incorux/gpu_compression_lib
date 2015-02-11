#ifndef MEMORY_CUH_AKYD86EK
#define MEMORY_CUH_AKYD86EK 1

#include "tools.cuh"
#include <stdio.h>
#include <list>
// Memory allocation manager
typedef struct allocation_info
{
    void *data;
    unsigned long size;
    char device;
    bool freed;
} allocation_info;

typedef std::list<struct allocation_info> mmManager;

void mmCudaMallocHost ( mmManager &manager, void **data, unsigned long size);
void mmCudaMalloc     ( mmManager &manager, void **data, unsigned long size);
void mmCudaFreeAll    ( mmManager &manager);
void mmCudaFree       ( mmManager &manager, void *ptr);

#endif /* end of include guard: MEMORY_CUH_AKYD86EK */

#include "memory.cuh"

void mmCudaMallocHost(mmManager &manager, void **data, unsigned long size)
{
    gpuErrchk(cudaMallocHost(data, size));
    allocation_info el;// = {.data = *data, .size = size, .device = 0}; -- clang warns about this construction -- strange behavior 
    el.data = *data;
    el.size = size;
    el.device = 0;
    el.freed = false;
    manager.push_front(el);
}
void mmCudaMalloc(mmManager &manager, void **data, unsigned long size)
{
    gpuErrchk(cudaMalloc(data, size));
    allocation_info el;// = {.data = *data, .size = size, .device = 1};
    el.data = *data;
    el.size = size;
    el.device = 1;
    el.freed = false;
    manager.push_front(el);
}

void __mmCudaFreeInternal(allocation_info *d)
{
    if (!(*d).freed) {
        if ((*d).device) { 
            gpuErrchk(cudaFree((*d).data));
        } else {
            gpuErrchk(cudaFreeHost((*d).data));
        }
        (*d).freed = true;
    }
}

void mmCudaFreeAll(mmManager &manager)
{
   mmManager::iterator i;
   for(i=manager.begin(); i != manager.end(); ++i) 
       __mmCudaFreeInternal(&(*i));
   manager.clear();
}

void mmCudaFree(mmManager &manager, void *ptr)
{
   mmManager::iterator i;
   for(i=manager.begin(); i != manager.end(); ++i) 
       if ((*i).data == ptr) __mmCudaFreeInternal(&(*i));
}

void _cudaErrorCheck(const char *file, int line)
{
    gpuAssert( cudaPeekAtLastError(), file, line );
    gpuAssert( cudaDeviceSynchronize(), file, line );
}
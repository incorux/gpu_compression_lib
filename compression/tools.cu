#include "tools.cuh"

void tiStart(tiManager &manager)
{
    timeit_info *el = (timeit_info *) malloc (sizeof(timeit_info));
    memset(el, 0 , sizeof(*el));
    
    cudaEventCreate( &(el->__start) ); 
    cudaEventCreate( &(el->__stop) ); 
    cudaEventRecord( el->__start, 0 );

    manager.push_front(el);
}

static unsigned long __xorshf96_x=123456789, __xorshf96_y=362436069, __xorshf96_z=521288629;

unsigned long xorshf96(void) {          //period 2^96-1
    unsigned long t;
        __xorshf96_x ^= __xorshf96_x << 16;
        __xorshf96_x ^= __xorshf96_x >> 5;
        __xorshf96_x ^= __xorshf96_x << 1;

        t = __xorshf96_x;
        __xorshf96_x = __xorshf96_y;
        __xorshf96_y = __xorshf96_z;
        __xorshf96_z = t ^ __xorshf96_x ^ __xorshf96_y;

        return __xorshf96_z;
}
// This is only for test purposes so it is optimized for speed (true randomness is not needed)
void big_random_block( unsigned long size, int limit_bits, int *data) 
{
    unsigned int mask = NBITSTOMASK(limit_bits);
    for (unsigned long i = 0; i < size; i++)
        data[i] = xorshf96() & mask;
}

void tiEnd(tiManager &manager, const char * name)
{
    timeit_info *el = manager.front();

    cudaEventRecord( el->__stop, 0 );
    cudaEventSynchronize( el->__stop );
    cudaEventElapsedTime( &(el->__elapsedTime), el->__start, el->__stop );
    
    el->name = strdup(name);
}
void tiPreatyPrint(tiManager &manager)
{
   tiManager::iterator i;
   float sum = 0.0;
   for(i=manager.begin(); i != manager.end(); ++i)
       sum += (*i)->__elapsedTime;
   printf("Elapsed time: %f [ms] (", sum);
   for(i=manager.begin(); i != manager.end(); ++i)
       printf("%s %f, ", (*i)->name, (*i)->__elapsedTime );
   printf(")\n");
}

void tiPreatyPrintThrougput(tiManager &manager, int data_size)
{
   tiManager::iterator i;
   float sum = 0.0;
   int gb = 1024 * 1024 * 1024, sec=1000;

   // Print sum
   for(i=manager.begin(); i != manager.end(); ++i)
       sum += (*i)->__elapsedTime;
   printf("Time, %.2f,ms, ", sum);

   // Print sum for operations marked as *
   sum = 0.0;
   for(i=manager.begin(); i != manager.end(); ++i)
       if ((*i)->name[0] == '*')
           sum += (*i)->__elapsedTime;
   printf("*, %.2f,ms, %.2f,GB/s, ", sum, ((float)data_size / gb)/ (sum/sec));

   // Print each operation
   for(i=manager.begin(); i != manager.end(); ++i)
       printf("%s, %.2f,ms, %.2f,GB/s, ", (*i)->name, (*i)->__elapsedTime, ((float)data_size / gb) / ((float)(*i)->__elapsedTime / sec) );
   printf("\n");
}


void tiClear(tiManager &manager)
{
   tiManager::iterator i;
   for(i=manager.begin(); i != manager.end(); ++i) {
       free(*i);
   }
   manager.clear();
}

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

int compare_arrays(int *in1, int *in2, unsigned long size)
{
    unsigned long count_errors = 0;
    for(unsigned long i = 0; i < size; i++) {
        if(in1[i] != in2[i]) {
            DPRINT(("Error at %ld element (%d != %d)\n ", i, in1[i], in2[i]));
            count_errors += 1;
        }
    }
    if (count_errors)
        DPRINT(("<================== ERROR ============= size = %ld errors = %ld\n", size, count_errors));
    return count_errors;
}

int compare_arrays_float(float *in1, float *in2, unsigned long size)
{
    int count_errors = 0;
    for(unsigned long i = 0; i < size; i++) {
        if(in1[i] != in2[i]) {
            DPRINT(("Error at %ld element (%f != %f)\n ", i, in1[i], in2[i]));
            count_errors += 1;
        }
    }
    if (count_errors)
        DPRINT(("<================== ERROR ============= size = %ld errors = %d\n", size, count_errors));
    return count_errors;
}

void _cudaErrorCheck(const char *file, int line)
{
    gpuAssert( cudaPeekAtLastError(), file, line );
    gpuAssert( cudaDeviceSynchronize(), file, line );
}

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

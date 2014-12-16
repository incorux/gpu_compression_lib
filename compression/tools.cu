#include "tools.cuh"
__device__ __host__ int bitLen(int a)
{
    int l = 1;
    while( (a = a>>1) ) l ++;
    return l;
}

__host__ __device__
int bitLen2( int v)
{
    register unsigned int r; // result of log2(v) will go here
    register unsigned int shift;

    r =     (v > 0xFFFF) << 4; v >>= r;
    shift = (v > 0xFF  ) << 3; v >>= shift; r |= shift;
    shift = (v > 0xF   ) << 2; v >>= shift; r |= shift;
    shift = (v > 0x3   ) << 1; v >>= shift; r |= shift;
    r |= (v >> 1);
    return r+1;
}

void tiStart(tiManager &manager)
{
    timeit_info *el = (timeit_info *) malloc (sizeof(timeit_info));
    memset(el, 0 , sizeof(*el));
    
    cudaEventCreate( &(el->__start) ); 
    cudaEventCreate( &(el->__stop) ); 
    cudaEventRecord( el->__start, 0 );

    manager.push_front(el);
}

void tiEnd(tiManager &manager, char * name)
{
    timeit_info *el = manager.front();

    cudaEventRecord( el->__stop, 0 );
    cudaEventSynchronize( el->__stop );
    cudaEventElapsedTime( &(el->__elapsedTime), el->__start, el->__stop );
    
    el->name = name;
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
   for(i=manager.begin(); i != manager.end(); ++i)
       sum += (*i)->__elapsedTime;
   printf("Time, %.2f, ms, ", sum);

   sum = 0.0;
   for(i=manager.begin(); i != manager.end(); ++i)
       if ((*i)->name[0] == '*')
           sum += (*i)->__elapsedTime;
   printf("Oper *, %.2f, ms, %.2f, GB/s, ", sum, ((float)data_size / gb)/ (sum/sec));

   for(i=manager.begin(); i != manager.end(); ++i)
       printf("%s, %.2f, ms, %.2f, GB/s, ", (*i)->name, (*i)->__elapsedTime, ((float)data_size / gb) / ((float)(*i)->__elapsedTime / sec) );
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

void mmCudaMallocHost(mmManager &manager, void **data, size_t size)
{
    gpuErrchk(cudaMallocHost(data, size));
    allocation_info el;// = {.data = *data, .size = size, .device = 0}; -- clang warns about this construction -- strange behavior 
    el.data = *data;
    el.size = size;
    el.device = 0;
    manager.push_front(el);
}
void mmCudaMalloc(mmManager &manager, void **data, size_t size)
{
    gpuErrchk(cudaMalloc(data, size));
    allocation_info el;// = {.data = *data, .size = size, .device = 1};
    el.data = *data;
    el.size = size;
    el.device = 1;
    manager.push_front(el);
}
void mmCudaFreeAll(mmManager &manager)
{
   mmManager::iterator i;
   for(i=manager.begin(); i != manager.end(); ++i) 
   {
       if ((*i).device) { 
           gpuErrchk(cudaFree((*i).data));
       } else {
           gpuErrchk(cudaFreeHost((*i).data));
       }
   }
   manager.clear();
}

int compare_arrays(int *in1, int *in2, size_t size)
{
    size_t count_errors = 0;
    for(size_t i = 0; i < size; i++) {
        if(in1[i] != in2[i]) {
            printf("Error at %ld element (%d != %d)\n ", i, in1[i], in2[i]);
            count_errors += 1;
        }
    }
    if (count_errors)
        DPRINT(("<================== ERROR ============= size = %ld errors = %ld\n", size, count_errors));
    return count_errors;
}

int compare_arrays_float(float *in1, float *in2, size_t size)
{
    int count_errors = 0;
    for(size_t i = 0; i < size; i++) {
        if(in1[i] != in2[i]) {
            DPRINT(("Error at %ld element (%f != %f)\n ", i, in1[i], in2[i]));
            count_errors += 1;
        }
    }
    if (count_errors)
        DPRINT(("<================== ERROR ============= size = %ld errors = %d\n", size, count_errors));
    return count_errors;
}

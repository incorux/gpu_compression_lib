#include "tools/tools.cuh"

#include "compression/afl_gpu.cuh"
#include <cuda.h>
#include <stdio.h>
#include <math.h>

int gpuid_0=1, gpuid_1=2;

#define PPRINT_THROUGPUT(name, data_size) printf("%c[1;34m",27);  printf name; printf("%c[30m,%c[37m ", 27,27); TIMEIT_PRINT_THROUGPUT(data_size);

template <typename T>
__global__ void saxpy(unsigned long n, int a, T *x, T *y)
{    
    // Determine element to process from thread index    
    for (long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) 
        y[tid] += a*x[tid];
}

template <typename T>
void multi_gpu_compress(unsigned long max_size, unsigned int bit_length, bool direct_copy)
{
    mmManager manager;
    T *dev0_data, *dev0_comp_out;
    T *dev1_data, *dev1_data_out, *dev1_comp_out;

    T *host_data, *host_data2;

    unsigned long comp_size = ((max_size * bit_length)/(sizeof(T)*8) + (sizeof(T)*8)) * sizeof(T);

    unsigned long data_size = max_size * sizeof(T); 

    TIMEIT_SETUP();

    mmCudaMallocHost(manager, (void**)&host_data,  max_size * sizeof(T));
    mmCudaMallocHost(manager, (void**)&host_data2, max_size * sizeof(T));

    big_random_block(max_size, bit_length, host_data);

    gpuErrchk(cudaSetDevice(gpuid_0));
    mmCudaMalloc(manager, (void **) &dev0_data, max_size * sizeof(T));
    mmCudaMalloc(manager, (void **) &dev0_comp_out, comp_size);

    TIMEIT_START();
    gpuErrchk( cudaMemcpy(dev0_data, host_data, data_size, cudaMemcpyHostToDevice) );
    TIMEIT_END("M->G");

    gpuErrchk(cudaSetDevice(gpuid_1));
    mmCudaMalloc(manager, (void **) &dev1_data, max_size * sizeof(T));
    mmCudaMalloc(manager, (void **) &dev1_comp_out, comp_size);

    gpuErrchk(cudaSetDevice(gpuid_0));


    TIMEIT_START();
    run_afl_compress_gpu <T, FL_ALGORITHM_MOD_AFL> (bit_length, dev0_data, dev0_comp_out, max_size);
    cudaErrorCheck();
    TIMEIT_END("*C");

    T *dev_data_source = dev0_comp_out;
    gpuErrchk(cudaSetDevice(gpuid_1));

    if (direct_copy)
    {
        TIMEIT_START();
        cudaMemcpyPeer(dev1_comp_out, gpuid_1, dev0_comp_out, gpuid_0, comp_size);
        TIMEIT_END("*copy");
        dev_data_source = dev1_comp_out;
        cudaErrorCheck();
    }

    TIMEIT_START();
    run_afl_decompress_gpu <T, FL_ALGORITHM_MOD_AFL> (bit_length, dev_data_source, dev1_data, max_size);
    cudaErrorCheck();
    TIMEIT_END("*D");


    mmCudaFree(manager, dev_data_source);
    mmCudaMalloc(manager, (void **) &dev1_data_out, max_size * sizeof(T));
    
    TIMEIT_START();
    saxpy <<<4096, 512>>> (max_size, 10, dev1_data, dev1_data_out);
    cudaErrorCheck();
    TIMEIT_END("saxpy");

    cudaMemset(host_data2, 0, data_size); 
    TIMEIT_START();
    gpuErrchk(cudaMemcpy(host_data2, dev1_data, data_size, cudaMemcpyDeviceToHost));
    TIMEIT_END("G->M");

    compare_arrays(host_data2, host_data, max_size);
    
    PPRINT_THROUGPUT(("MGPU%s compr afl%d", direct_copy ? "copy":"access", bit_length), max_size * sizeof(T));

    mmCudaFreeAll(manager);
}


template <typename T>
void multi_gpu(unsigned long max_size, bool direct_copy)
{
    mmManager manager;
    T *dev0_data, *dev1_data;
    T *dev1_data_out;

    T *host_data, *host_data2;

    unsigned long data_size = max_size * sizeof(T); 

    gpuErrchk(cudaSetDevice(gpuid_0));
    mmCudaMalloc(manager, (void **) &dev0_data, max_size * sizeof(T));

    gpuErrchk(cudaSetDevice(gpuid_1));
    mmCudaMalloc(manager, (void **) &dev1_data, max_size * sizeof(T));
    mmCudaMalloc(manager, (void **) &dev1_data_out, max_size * sizeof(T));

    mmCudaMallocHost(manager, (void**)&host_data,  max_size * sizeof(T));
    mmCudaMallocHost(manager, (void**)&host_data2, max_size * sizeof(T));

    big_random_block(max_size, 31, host_data); // We do not compress this so any bitlen is OK

    TIMEIT_SETUP();

    TIMEIT_START();
    gpuErrchk( cudaMemcpy(dev0_data, host_data, data_size, cudaMemcpyHostToDevice) );
    TIMEIT_END("M->G");

    T *dev_data_source = dev0_data;

    if (direct_copy)
    {
        TIMEIT_START();
        /*cudaMemcpy(dev1_data, dev0_data, max_size * sizeof(T), cudaMemcpyDefault);*/
        cudaMemcpyPeer(dev1_data, gpuid_1, dev0_data, gpuid_0, max_size * sizeof(T));
        TIMEIT_END("*copy");
        dev_data_source = dev1_data;
        cudaErrorCheck();
    }
    
    TIMEIT_START();
    saxpy <<<4096, 512>>> (max_size, 10, dev_data_source, dev1_data_out);
    cudaErrorCheck();
    TIMEIT_END("saxpy");

    cudaMemset(host_data2, 0, data_size); 
    TIMEIT_START();
    gpuErrchk(cudaMemcpy(host_data2, dev_data_source, data_size, cudaMemcpyDeviceToHost));
    TIMEIT_END("G->M");

    compare_arrays(host_data2, host_data, max_size);
    
    PPRINT_THROUGPUT(("MGPU%s", direct_copy ? "copy":"access"), max_size * sizeof(T));

    mmCudaFreeAll(manager);
}

int main(int argc, char *argv[])
{

    unsigned long max_size = 10000000;
    printf("%s [size] [dev0_id, dev1_id]\n", argv[0]);
    if(argc > 1) {
        if ( atol(argv[1]))
            max_size = atol(argv[1]);

        if (argc == 4) {
            gpuid_0 = atoi(argv[2]);
            gpuid_1 = atoi(argv[3]);
        }
    }

    printf("Data size: %ld,using device %d and device %d\n", max_size, gpuid_0, gpuid_1 );


    int can_access_peer_0_1, can_access_peer_1_0;
    gpuErrchk(cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_0, gpuid_1));
    gpuErrchk(cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpuid_1, gpuid_0));
    printf("can acces device %d->%d: %d  %d->%d %d\n",gpuid_0, gpuid_1, can_access_peer_0_1, gpuid_1, gpuid_0, can_access_peer_1_0 );

    gpuErrchk(cudaSetDevice(gpuid_0));
    gpuErrchk(cudaDeviceEnablePeerAccess(gpuid_1, 0));

    gpuErrchk(cudaSetDevice(gpuid_1));
    gpuErrchk(cudaDeviceEnablePeerAccess(gpuid_0, 0));


    multi_gpu <int> (max_size, true);
    multi_gpu <int> (max_size, false);

    multi_gpu <long> (max_size, true);
    multi_gpu <long> (max_size, false);

    for (int i = 2; i < 32; ++i)
    {
        multi_gpu_compress <int> (max_size, i, true);
        multi_gpu_compress <int> (max_size, i, false);
    }

    for (int i = 32; i < 64; ++i)
    {
        multi_gpu_compress <long> (max_size, i, true);
        multi_gpu_compress <long> (max_size, i, false);
    }
    
    return 0;
}

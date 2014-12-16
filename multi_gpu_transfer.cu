#include "compression/tools.cuh"
#include "compression/avar_gpu.cuh"
#include <cuda.h>
#include <stdio.h>
#include <math.h>

int gpuid_0=1, gpuid_1=2;

#define PPRINT_THROUGPUT(name, data_size) printf("%c[1;34m",27);  printf name; printf("%c[30m,%c[37m ", 27,27); TIMEIT_PRINT_THROUGPUT(data_size);

__global__ void saxpy(size_t n, int a, int *x, int *y)
{    
    // Determine element to process from thread index    
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) 
        y[tid] += a*x[tid];
}

void multi_gpu_compress(size_t max_size, unsigned int bit_length, bool direct_copy)
{
    mmManager manager;
    int *dev0_data, *dev0_comp_out;
    int *dev1_data, *dev1_data_out, *dev1_comp_out;

    long comp_size = ((max_size * bit_length)/32 +32) * sizeof(int);


    gpuErrchk(cudaSetDevice(gpuid_0));
    mmCudaMalloc(manager, (void **) &dev0_data, max_size * sizeof(int));
    mmCudaMalloc(manager, (void **) &dev0_comp_out, comp_size);

    gpuErrchk(cudaSetDevice(gpuid_1));
    mmCudaMalloc(manager, (void **) &dev1_data, max_size * sizeof(int));
    mmCudaMalloc(manager, (void **) &dev1_data_out, max_size * sizeof(int));
    mmCudaMalloc(manager, (void **) &dev1_comp_out, comp_size);

    gpuErrchk(cudaSetDevice(gpuid_0));
    avar_header comp_h = { bit_length } ;

    TIMEIT_SETUP();

    TIMEIT_START();
    run_avar_compress_gpu(comp_h, dev0_data, dev0_comp_out, max_size);
    TIMEIT_END("*C");
    cudaErrorCheck();

    int *dev_data_source = dev0_comp_out;
    gpuErrchk(cudaSetDevice(gpuid_1));

    if (direct_copy)
    {
        TIMEIT_START();
        cudaMemcpyPeerAsync(dev1_comp_out, gpuid_1, dev0_comp_out, gpuid_0, comp_size);
        cudaDeviceSynchronize();
        TIMEIT_END("*copy");
        dev_data_source = dev1_comp_out;
        cudaErrorCheck();
    }

    TIMEIT_START();
    run_avar_decompress_gpu(comp_h, dev_data_source, dev1_data, max_size);
    TIMEIT_END("*D");
    cudaErrorCheck();
    
    TIMEIT_START();
    saxpy <<<4096, 512>>> (max_size, 10, dev1_data, dev1_data_out);
    cudaErrorCheck();
    TIMEIT_END("saxpy");
    
    PPRINT_THROUGPUT(("MGPU%s compr avar%d", direct_copy ? "copy":"access", bit_length), max_size * sizeof(int));

    mmCudaFreeAll(manager);
}


void multi_gpu(size_t max_size, bool direct_copy)
{
    mmManager manager;
    int *dev0_data, *dev1_data;
    int *dev1_data_out;

    gpuErrchk(cudaSetDevice(gpuid_0));
    mmCudaMalloc(manager, (void **) &dev0_data, max_size * sizeof(int));

    gpuErrchk(cudaSetDevice(gpuid_1));
    mmCudaMalloc(manager, (void **) &dev1_data, max_size * sizeof(int));
    mmCudaMalloc(manager, (void **) &dev1_data_out, max_size * sizeof(int));

    TIMEIT_SETUP();

    int *dev_data_source = dev0_data;

    if (direct_copy)
    {
        TIMEIT_START();
        /*cudaMemcpy(dev1_data, dev0_data, max_size * sizeof(int), cudaMemcpyDefault);*/
        cudaMemcpyPeerAsync(dev1_data, gpuid_1, dev0_data, gpuid_0, max_size * sizeof(int));
        cudaDeviceSynchronize();
        TIMEIT_END("*copy");
        dev_data_source = dev1_data;
    }
    
    TIMEIT_START();
    saxpy <<<4096, 512>>> (max_size, 10, dev_data_source, dev1_data_out);
    cudaErrorCheck();
    TIMEIT_END("saxpy");
    
    PPRINT_THROUGPUT(("MGPU%s", direct_copy ? "copy":"access"), max_size * sizeof(int));

    mmCudaFreeAll(manager);
}

int main(int argc, char *argv[])
{

    size_t max_size = 10000000;
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
    printf("can acces device 0->1: %d  1->0 %d\n",can_access_peer_0_1, can_access_peer_1_0 );

    gpuErrchk(cudaSetDevice(gpuid_0));
    gpuErrchk(cudaDeviceEnablePeerAccess(gpuid_1, 0));

    gpuErrchk(cudaSetDevice(gpuid_1));
    gpuErrchk(cudaDeviceEnablePeerAccess(gpuid_0, 0));


    multi_gpu(max_size, true);
    multi_gpu(max_size, false);

    for (int i = 2; i < 32; ++i)
    {
        multi_gpu_compress(max_size, i, true);
        multi_gpu_compress(max_size, i, false);
    }
    
    return 0;
}

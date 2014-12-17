#include "compression/tools.cuh"
#include "compression/avar_gpu.cuh"

#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define PPRINT(name) printf("%c[1;34m",27);  printf name; printf("%c[30m Status: %c[1;32mCORRECT%c[37m ", 27,27,27); TIMEIT_PRINT();
/*#define PPRINT(name) printf("%c[1;34m",27);  printf name; printf("%c[30m Status: %c[1;32mOK%c[37m \n", 27,27,27);*/
#define PPRINT_MANY(name) printf("%c[1;34m",27);  printf name; printf("%c[30m: %c[1;32mOK%c[37m ", 27,27,27);

#define PPRINT_THROUGPUT(name, data_size) printf("%c[1;34m",27);  printf name; printf("%c[30m, %c[1;32mOK%c[37m, ", 27,27,27); TIMEIT_PRINT_THROUGPUT(data_size);

void avar_gpu_test(size_t max_size)
{
    int *dev_out;
    int *dev_data;
    int *host_data, *host_data2;

    mmManager manager;

    TIMEIT_SETUP();

    mmCudaMallocHost(manager,(void**)&host_data, max_size * sizeof(int));
    mmCudaMallocHost(manager,(void**)&host_data2, max_size * sizeof(int));

    mmCudaMalloc(manager, (void **) &dev_out, max_size * sizeof(int)); // maximal compression size
    mmCudaMalloc(manager, (void **) &dev_data, max_size * sizeof(int));

    for (unsigned int i = 2; i <= 31; ++i) {
        big_random_block(max_size, pow((double)2,(double)(i-1))-1, host_data);

        TIMEIT_START();
        gpuErrchk( cudaMemcpy(dev_data, host_data, max_size * sizeof(int), cudaMemcpyHostToDevice) );
        TIMEIT_END("M->G");

        avar_header comp_h = { i };
        cudaMemset(dev_out, 0, max_size * sizeof(int)); // Clean up before compression

        TIMEIT_START();
        run_avar_compress_gpu(comp_h, dev_data, dev_out, max_size);
        TIMEIT_END("*comp");
        cudaErrorCheck();

        cudaMemset(dev_data, 0, max_size * sizeof(int)); // Clean up before decompression

        TIMEIT_START();
        run_avar_decompress_gpu(comp_h, dev_out, dev_data, max_size);
        TIMEIT_END("*decomp");
        cudaErrorCheck();

        cudaMemset(host_data2, 0, max_size * sizeof(int)); // Clean up before compression
        TIMEIT_START();
        gpuErrchk(cudaMemcpy(host_data2, dev_data, max_size * sizeof(int), cudaMemcpyDeviceToHost));
        TIMEIT_END("G->M");

        compare_arrays(host_data2, host_data, max_size);
        PPRINT_THROUGPUT(("GPU avar%d", i), max_size * sizeof(int));

    }

    mmCudaFreeAll(manager);
}

int main(int argc, char *argv[])
{
    size_t max_size = 100000000;

    if (argc > 1 && atol(argv[1]))
        max_size = atol(argv[1]);

    printf("Data size: %ld\n", max_size );
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        avar_gpu_test(max_size);
    }
    return 0;
}
/* vim: set fdm=syntax: */

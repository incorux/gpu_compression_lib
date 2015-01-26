#include "compression/tools.cuh"
#include "compression/avar_gpu.cuh"
#include "compression/pavar_gpu.cuh"

#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define PPRINT(name) printf("%c[1;34m",27);  printf name; printf("%c[30m Status: %c[1;32mCORRECT%c[37m ", 27,27,27); TIMEIT_PRINT();
/*#define PPRINT(name) printf("%c[1;34m",27);  printf name; printf("%c[30m Status: %c[1;32mOK%c[37m \n", 27,27,27);*/
#define PPRINT_MANY(name) printf("%c[1;34m",27);  printf name; printf("%c[30m: %c[1;32mOK%c[37m ", 27,27,27);

#define PPRINT_THROUGPUT(name, data_size) printf("%c[1;34m",27);  printf name; printf("%c[30m, %c[1;32mOK%c[37m, ", 27,27,27); TIMEIT_PRINT_THROUGPUT(data_size);

void avar_gpu_test(unsigned long max_size)
{
    int *dev_out;
    int *dev_data;
    int *host_data, *host_data2;

    // for size less then 32 we actually will need more space than original data
    int compressed_data_size = (max_size < 32 ? 32 : max_size) * sizeof(int); 

    int data_size = max_size * sizeof(int); 

    mmManager manager;

    TIMEIT_SETUP();

    mmCudaMallocHost(manager, (void**)&host_data,  data_size);
    mmCudaMallocHost(manager, (void**)&host_data2, data_size);

    mmCudaMalloc(manager, (void **) &dev_out, compressed_data_size); 
    mmCudaMalloc(manager, (void **) &dev_data, data_size);

    for (unsigned int i = 2; i <= 31; ++i) {
        big_random_block(max_size, i, host_data);

        TIMEIT_START();
        gpuErrchk( cudaMemcpy(dev_data, host_data, data_size, cudaMemcpyHostToDevice) );
        TIMEIT_END("M->G");

        avar_header comp_h = { i };
        cudaMemset(dev_out, 0, compressed_data_size); // Clean up before compression

        TIMEIT_START();
        run_avar_compress_gpu(comp_h, dev_data, dev_out, max_size);
        TIMEIT_END("*comp");
        cudaErrorCheck();

        cudaMemset(dev_data, 0, data_size); // Clean up before decompression

        TIMEIT_START();
        run_avar_decompress_gpu(comp_h, dev_out, dev_data, max_size);
        TIMEIT_END("*decomp");
        cudaErrorCheck();

        cudaMemset(host_data2, 0, data_size); 
        TIMEIT_START();
        gpuErrchk(cudaMemcpy(host_data2, dev_data, data_size, cudaMemcpyDeviceToHost));
        TIMEIT_END("G->M");

        compare_arrays(host_data2, host_data, max_size);

        PPRINT_THROUGPUT(("GPU avar%d", i), data_size);
    }

    mmCudaFreeAll(manager);
}

void pavar_gpu_test(unsigned long max_size)
{
    int *dev_out;
    int *dev_data;
    int *host_data, *host_data2;
    int *dev_data_patch_index, *dev_data_patch_values, *dev_data_patch_count;
    int *dev_queue_patch_index, *dev_queue_patch_values, *dev_queue_patch_count;
    int outlier_count = 0.2 * max_size;

    mmManager manager;
    TIMEIT_SETUP();

    mmCudaMallocHost(manager,(void**)&host_data, max_size * sizeof(int));
    mmCudaMallocHost(manager,(void**)&host_data2, max_size * sizeof(int));

    mmCudaMalloc(manager, (void **) &dev_out, max_size * sizeof(int)); // maximal compression size
    mmCudaMalloc(manager, (void **) &dev_data, max_size * sizeof(int));
    
    mmCudaMalloc(manager, (void **) &dev_data_patch_count, sizeof(int));
    mmCudaMalloc(manager, (void **) &dev_data_patch_index, outlier_count * sizeof(int));
    mmCudaMalloc(manager, (void **) &dev_data_patch_values, outlier_count * sizeof(int));

    mmCudaMalloc(manager, (void **) &dev_queue_patch_count, sizeof(int));
    mmCudaMalloc(manager, (void **) &dev_queue_patch_index, outlier_count * sizeof(int));
    mmCudaMalloc(manager, (void **) &dev_queue_patch_values, outlier_count * sizeof(int));

    for (unsigned int i = 2; i <= 31; ++i) {
        big_random_block_with_outliers(max_size, outlier_count, i, i + 3, host_data);

        TIMEIT_START();
        gpuErrchk( cudaMemcpy(dev_data, host_data, max_size * sizeof(int), cudaMemcpyHostToDevice) );
        TIMEIT_END("M->G");

        pavar_header comp_h = { i, 4 };
        cudaMemset(dev_out, 0, max_size * sizeof(int)); // Clean up before compression
        cudaMemset(dev_data_patch_count, 0, sizeof(int)); // Clean up before compression
        cudaMemset(dev_queue_patch_count, 0, sizeof(int)); // Clean up before compression

        TIMEIT_START();
        //TODO
        run_pavar_compress_gpu(
                comp_h, 
                dev_data, 
                dev_out, 
                max_size,
                
                dev_queue_patch_values, 
                dev_queue_patch_index,
                dev_queue_patch_count,

                dev_data_patch_values,
                dev_data_patch_index,
                dev_data_patch_count
                );

        TIMEIT_END("*comp");
        cudaErrorCheck();

        cudaMemset(dev_data, 0, max_size * sizeof(int)); // Clean up before decompression

        TIMEIT_START();
        //TODO
        /*run_pavar_decompress_gpu(comp_h, dev_out, dev_data, max_size);*/
        TIMEIT_END("*decomp");
        cudaErrorCheck();

        /*cudaMemset(host_data2, 0, max_size * sizeof(int)); */
        TIMEIT_START();
        /*gpuErrchk(cudaMemcpy(host_data2, dev_data, max_size * sizeof(int), cudaMemcpyDeviceToHost));*/
        TIMEIT_END("G->M");

        /*compare_arrays(host_data2, host_data, max_size);*/

        PPRINT_THROUGPUT(("GPU pavar%d", i), max_size * sizeof(int));
    }

    mmCudaFreeAll(manager);
}

int main(int argc, char *argv[])
{
    unsigned long max_size = 100000000;

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
        /*pavar_gpu_test(max_size);*/
    }
    return 0;
}
/* vim: set fdm=syntax: */

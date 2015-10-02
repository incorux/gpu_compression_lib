#ifndef TEST_AFL_CUH_VSFESWCR
#define TEST_AFL_CUH_VSFESWCR

#include "catch.hpp"

#include "tools/tools.cuh"

#include "compression/afl_gpu.cuh"
#include "compression/pafl_gpu.cuh"

#include <typeinfo>

#define PPRINT_THROUGPUT(name, data_size) {printf("%c[1;34m",27);  printf name; printf("%c[30m, %c[37m", 27,27); TIMEIT_PRINT_THROUGPUT(data_size);}

template <typename T, int CWARP_SIZE>
class test_afl
{
public:
   virtual void allocateMemory() {
        mmCudaMallocHost(manager, (void**)&host_data,  data_size);
        mmCudaMallocHost(manager, (void**)&host_data2, data_size);

        mmCudaMalloc(manager, (void **) &dev_out, compressed_data_size); 
        mmCudaMalloc(manager, (void **) &dev_data, data_size);
    }

    virtual void initializeData(int bit_length) {
        big_random_block(max_size, bit_length, host_data);
    }

    virtual void transferDataToGPU() {
        gpuErrchk( cudaMemcpy(dev_data, host_data, data_size, cudaMemcpyHostToDevice) );
    }

    virtual void cleanBeforeCompress() {
        cudaMemset(dev_out, 0, compressed_data_size); // Clean up before compression
    }

    virtual void compressData(int bit_length) {
        run_afl_compress_gpu <T, CWARP_SIZE> (bit_length, dev_data, dev_out, max_size);
    }

    virtual void errorCheck() { 
        cudaErrorCheck();
    }

    virtual void cleanBeforeDecompress() {
        cudaMemset(dev_data, 0, data_size); // Clean up before decompression
    }

    virtual void decompressData(int bit_length) {
        run_afl_decompress_gpu <T, CWARP_SIZE> (bit_length, dev_out, dev_data, max_size);
    }

    virtual void transferDataFromGPU() {
        cudaMemset(host_data2, 0, data_size); 
        gpuErrchk(cudaMemcpy(host_data2, dev_data, data_size, cudaMemcpyDeviceToHost));
    }

    virtual void setup(unsigned long max_size) {
        this->max_size = max_size;
        this->cword = sizeof(T) * 8;
        this->data_size = max_size * sizeof(T);
        // for size less then cword we actually will need more space than original data
        this->compressed_data_size = (max_size < cword  ? cword : max_size) * sizeof(T);
    }

    virtual void run(unsigned int max_size, bool print = false)
    {
        setup(max_size);

        allocateMemory();
        TIMEIT_SETUP();

        for (unsigned int bit_length = 1; bit_length < cword; ++bit_length) {
            initializeData(bit_length);

            TIMEIT_START();
            transferDataToGPU();
            TIMEIT_END("M->G");
            
            cleanBeforeCompress();
            
            TIMEIT_START();
            compressData(bit_length);
            TIMEIT_END("*comp");
            
            errorCheck();

            cleanBeforeDecompress();

            TIMEIT_START();
            decompressData(bit_length);
            TIMEIT_END("*comp");
            
            errorCheck();

            TIMEIT_START();
            transferDataFromGPU();
            TIMEIT_END("G->M");

            CAPTURE(bit_length);
            CHECK(testData()==0);
            
            if(print) PPRINT_THROUGPUT(("%s, %s, %d", __PRETTY_FUNCTION__, typeid(T).name(), bit_length), data_size);
        }
    }

    T testData() {
       return compare_arrays(host_data2, host_data, max_size);
    }

    virtual ~test_afl () {
        mmCudaFreeAll(manager);
    }

protected:
    T *dev_out;
    T *dev_data;
    T *host_data; 
    T *host_data2;

    unsigned int cword;

    unsigned int compressed_data_size;
    unsigned long data_size;
    unsigned long max_size;

    mmManager manager;
};


template <typename T, int CWARP_SIZE> class test_afl_random_access: public test_afl<T, CWARP_SIZE> {
public: virtual void decompressData(int bit_length) {
        run_afl_decompress_value_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data, this->max_size);
    }
};

template <typename T, int CWARP_SIZE> class test_afl_cpu: public test_afl<T, CWARP_SIZE> {
public: 

   virtual void allocateMemory() {
        mmCudaMallocHost(this->manager, (void**)&this->host_data,  this->data_size);
        mmCudaMallocHost(this->manager, (void**)&this->host_data2, this->data_size);

        mmCudaMallocHost(this->manager, (void **)&this->host_out, this->data_size); 
    }

    virtual void transferDataToGPU() {}
    virtual void cleanBeforeCompress() {} 
    virtual void errorCheck() {}
    virtual void cleanBeforeDecompress() {}
    virtual void transferDataFromGPU() {}

    virtual void compressData(int bit_length) {
        run_afl_compress_cpu <T, CWARP_SIZE> (bit_length, this->host_data, this->host_out, this->max_size);
    }

    virtual void decompressData(int bit_length) {
        run_afl_decompress_cpu <T, CWARP_SIZE> (bit_length, this->host_out, this->host_data2, this->max_size);
    }
protected:
        T *host_out;
};

template <typename T, int CWARP_SIZE> class test_afl_random_access_cpu: public test_afl<T, CWARP_SIZE> {
public: 

   virtual void allocateMemory() {
        mmCudaMallocHost(this->manager, (void**)&this->host_data,  this->data_size);
        mmCudaMallocHost(this->manager, (void**)&this->host_data2, this->data_size);

        mmCudaMallocHost(this->manager, (void **)&this->host_out, this->data_size); 
    }

    virtual void transferDataToGPU() {}
    virtual void cleanBeforeCompress() {} 
    virtual void errorCheck() {}
    virtual void cleanBeforeDecompress() {}
    virtual void transferDataFromGPU() {}

    virtual void compressData(int bit_length) {
        run_afl_compress_value_cpu <T, CWARP_SIZE> (bit_length, this->host_data, this->host_out, this->max_size);
    }

    virtual void decompressData(int bit_length) {
        run_afl_decompress_value_gpu <T, CWARP_SIZE> (bit_length, this->host_out, this->host_data2, this->max_size);
    }
protected:
        T *host_out;
};

#endif /* end of include guard: TEST_AFL_CUH_VSFESWCR */

#include "catch.hpp"

#include "tools/tools.cuh"

#include "compression/afl_gpu.cuh"
#include "compression/pafl_gpu.cuh"

#include <fstream>
#include <string>
#include <map>

#define PPRINT_THROUGPUT(name, data_size) {printf("%c[1;34m",27);  printf name; printf("%c[30m, %c[37m", 27,27); TIMEIT_PRINT_THROUGPUT(data_size);}

template <typename T, int CWARP_SIZE>
class test_afl
{
public:
    void allocateMemory() {
        mmCudaMallocHost(manager, (void**)&host_data,  data_size);
        mmCudaMallocHost(manager, (void**)&host_data2, data_size);

        mmCudaMalloc(manager, (void **) &dev_out, compressed_data_size); 
        mmCudaMalloc(manager, (void **) &dev_data, data_size);
    }

    void initializeData(int bit_length) {
        big_random_block(max_size, bit_length, host_data);
    }

    void transferDataToGPU() {
        gpuErrchk( cudaMemcpy(dev_data, host_data, data_size, cudaMemcpyHostToDevice) );
    }

    void cleanBeforeCompress() {
        cudaMemset(dev_out, 0, compressed_data_size); // Clean up before compression
    }

    virtual void compressData(int bit_length) {
        run_afl_compress_gpu <T, CWARP_SIZE> (bit_length, dev_data, dev_out, max_size);
    }

    void errorCheck() { 
        cudaErrorCheck();
    }

    void cleanBeforeDecompress() {
        cudaMemset(dev_data, 0, data_size); // Clean up before decompression
    }

    virtual void decompressData(int bit_length) {
        run_afl_decompress_gpu <T, CWARP_SIZE> (bit_length, dev_out, dev_data, max_size);
    }

    void transferDataFromGPU() {
        cudaMemset(host_data2, 0, data_size); 
        gpuErrchk(cudaMemcpy(host_data2, dev_data, data_size, cudaMemcpyDeviceToHost));
    }

    void run(unsigned int max_size, bool print = false)
    {
        this->max_size = max_size;
        cword = sizeof(T) * 8;
        data_size = max_size * sizeof(T);
        // for size less then cword we actually will need more space than original data
        compressed_data_size = (max_size < cword  ? cword : max_size) * sizeof(T);

        allocateMemory();
        TIMEIT_SETUP();

        for (unsigned int i = 1; i < cword; ++i) {
            initializeData(i);

            TIMEIT_START();
            transferDataToGPU();
            TIMEIT_END("M->G");
            
            cleanBeforeCompress();
            
            TIMEIT_START();
            compressData(i);
            TIMEIT_END("*comp");
            
            errorCheck();

            cleanBeforeDecompress();

            TIMEIT_START();
            decompressData(i);
            TIMEIT_END("*comp");
            
            errorCheck();

            TIMEIT_START();
            transferDataFromGPU();
            TIMEIT_END("G->M");

            CHECK(testData()==0);
            
            if(print) PPRINT_THROUGPUT(("%s fl=%d", __PRETTY_FUNCTION__, i), data_size);
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

    int cword;

    int compressed_data_size;
    unsigned long data_size;
    unsigned long max_size;

    mmManager manager;
};


template <typename T, int CWARP_SIZE> class test_afl_random_access: public test_afl<T, CWARP_SIZE> {
public: virtual void decompressData(int bit_length) {
        run_afl_decompress_value_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data, this->max_size);
    }
};

#define SMALL_DATA_SET 1000 
#define MEDIUM_DATA_SET  100000
#define PERF_DATA_SET  100000000

#define RUN_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " test set", "[" NAME "]" ) {\
    SECTION("int: SMALL data set")   {CNAME <int, PARAM> ().run(SMALL_DATA_SET);}\
    SECTION("int: MEDIUM data set")  {CNAME <int, PARAM>  ().run(MEDIUM_DATA_SET);}\
    SECTION("long: SMALL data set")  {CNAME <long, PARAM> ().run(SMALL_DATA_SET);}\
    SECTION("long: MEDIUM data set")  {CNAME <long, PARAM> ().run(MEDIUM_DATA_SET);}\
}

RUN_TEST("AFL", test_afl, 32);
RUN_TEST("FL", test_afl, 1);

RUN_TEST("RAFL", test_afl_random_access, 32);
RUN_TEST("RFL", test_afl_random_access, 1);

#define RUN_PERF_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " performance test", "[" NAME "][PERF][hide]" ) {\
    SECTION("int: PERF data set")   {CNAME <int, PARAM> ().run(PERF_DATA_SET, true);}\
    SECTION("long: PERF data set")  {CNAME <int, PARAM>  ().run(PERF_DATA_SET, true);}\
}

RUN_PERF_TEST("AFL", test_afl, 32);
RUN_PERF_TEST("FL", test_afl, 1);

RUN_PERF_TEST("RAFL", test_afl_random_access, 32);
RUN_PERF_TEST("RFL", test_afl_random_access, 1);



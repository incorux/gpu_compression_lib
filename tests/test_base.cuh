#ifndef TEST_BASE_CUH_WT3FRCI9
#define TEST_BASE_CUH_WT3FRCI9

#include "catch.hpp"
#include "tools/tools.cuh"
#include <typeinfo>
#include <string>

#define PPRINT_THROUGPUT(name, data_size) {printf("%c[1;34m",27);  printf name; printf("%c[30m; %c[37m", 27,27); TIMEIT_PRINT_THROUGPUT(data_size);}

template <typename T, int CWARP_SIZE>
class test_base
{
public:

   virtual void decompressData(int bit_length) = 0;
   virtual void compressData(int bit_length) = 0;

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


    virtual void errorCheck() { 
        cudaErrorCheck();
    }

    virtual void cleanBeforeDecompress() {
        cudaMemset(dev_data, 0, data_size); // Clean up before decompression
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
            TIMEIT_END("*decomp");
            
            errorCheck();

            TIMEIT_START();
            transferDataFromGPU();
            TIMEIT_END("G->M");

            CAPTURE(bit_length);
            CHECK(testData());
            
            if(print) PPRINT_THROUGPUT(("%s; %s; %d", __PRETTY_FUNCTION__, typeid(T).name(), bit_length), data_size);
        }
    }

    virtual T testData() {
       return compare_arrays(host_data2, host_data, max_size);
    }

    virtual ~test_base () {
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

#define RUN_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " test set", "[" NAME "]") {\
    SECTION("int: SMALL ALIGNED data set")   {CNAME <int, PARAM> ().run(SMALL_ALIGNED_DATA_SET);}\
    SECTION("int: SMALL data set")   {CNAME <int, PARAM> ().run(SMALL_DATA_SET);}\
    SECTION("int: MEDIUM data set")  {CNAME <int, PARAM>  ().run(MEDIUM_DATA_SET);}\
    SECTION("long: SMALL ALIGNED data set")  {CNAME <long, PARAM> ().run(SMALL_ALIGNED_DATA_SET);}\
    SECTION("long: SMALL data set")  {CNAME <long, PARAM> ().run(SMALL_DATA_SET);}\
    SECTION("long: MEDIUM data set")  {CNAME <long, PARAM> ().run(MEDIUM_DATA_SET);}\
}

#define RUN_PERF_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " performance test", "[" NAME "][PERF][hide]" ) {\
    SECTION("int: PERF data set")   {CNAME <int, PARAM> ().run(PERF_DATA_SET, true);}\
    SECTION("int: PERF data set")   {CNAME <long, PARAM> ().run(PERF_DATA_SET, true);}\
}

#define RUN_BENCHMARK_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " benchmark test", "[" NAME "][BENCHMARK][hide]" ) {\
    long i;\
    SECTION("int: BENCHMARK data set")   {\
        for (i = 1000; i < 1000000; i*=10)\
            CNAME <int, PARAM> ().run(i, true);\
        for (i = 1000000; i< 100000000; i+= 10 * 1000000)\
            CNAME <int, PARAM> ().run(i, true);\
        for (i = 100000000; i<= 200000000; i+= 5 * 10000000)\
            CNAME <int, PARAM> ().run(i, true);\
    }\
    SECTION("long: BENCHMARK data set")   {\
        for (i = 1000; i < 1000000; i*=10)\
            CNAME <long, PARAM> ().run(i, true);\
        for (i = 1000000; i< 100000000; i+= 10 * 1000000)\
            CNAME <long, PARAM> ().run(i, true);\
        for (i = 100000000; i<= 200000000; i+= 5 * 10000000)\
            CNAME <long, PARAM> ().run(i, true);\
    }\
}

#endif /* end of include guard: TEST_BASE_CUH_WT3FRCI9 */

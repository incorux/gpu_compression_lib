#ifndef TEST_BASE_CUH_WT3FRCI9
#define TEST_BASE_CUH_WT3FRCI9

#include "catch.hpp"
#include "config.cuh" 
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

    virtual void pre_setup(unsigned long max_size) {
        this->cword = sizeof(T) * 8;
        this->max_size = max_size;
        this->data_size = max_size * sizeof(T);
    }

    virtual void setup(unsigned long max_size) {
        unsigned long data_block_size = sizeof(T) * 8 * 32;

        // for size less then cword we actually will need more space than original data
        this->compressed_data_size = (max_size < data_block_size  ? data_block_size : max_size);
        this->compressed_data_size = ((this->compressed_data_size * this->bit_length + (32*sizeof(T)*8)-1) / (32*sizeof(T)*8)) * 32 * sizeof(T) + (sizeof(T)*8) * sizeof(T);

        if (if_debug()){
            printf("Comp ratio %f",  (double)((double)this->compressed_data_size / ( double)this->data_size));
            printf(" %d %lu %ld %ld\n" , this->bit_length, max_size, this->data_size, this->compressed_data_size);
        }
    }

    virtual int run(unsigned long max_size, bool print = false)
    {
        TIMEIT_SETUP();
        pre_setup(max_size);

        int error_count = 0;

        /* for (unsigned int _bit_lenght = 1; _bit_lenght < cword; ++_bit_lenght) { */
        //TODO
        for (unsigned int _bit_lenght = 23; _bit_lenght < cword; ++_bit_lenght) {
            this->bit_length = _bit_lenght;
            setup(max_size);

            allocateMemory();
            mmCudaReportUsage(manager);

            initializeData(_bit_lenght);

            TIMEIT_START();
            transferDataToGPU();
            TIMEIT_END("M->G");
            
            cleanBeforeCompress();
            errorCheck();
            
            TIMEIT_START();
            compressData(_bit_lenght);
            TIMEIT_END("*comp");
            
            errorCheck();

            cleanBeforeDecompress();

            TIMEIT_START();
            decompressData(_bit_lenght);
            TIMEIT_END("*decomp");
            
            errorCheck();

            TIMEIT_START();
            transferDataFromGPU();
            TIMEIT_END("G->M");

            if(testData()) { 
                printf("\n===== %d =====\n", _bit_lenght); 
                error_count +=1; 
            }
            
            if(print) PPRINT_THROUGPUT(("%s; %s; %d", __PRETTY_FUNCTION__, typeid(T).name(), _bit_lenght), data_size);

            mmCudaFreeAll(manager);
        }

        return error_count;
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

    unsigned long compressed_data_size;
    unsigned long data_size;
    unsigned long max_size;
    unsigned int bit_length;

    mmManager manager;
};

#define RUN_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " test set", "[" NAME "][ALL]") {\
    SECTION("int: SMALL ALIGNED data set")  { CNAME <int, PARAM> test  ; CHECK(test.run(SMALL_ALIGNED_DATA_SET) == 0 );}\
    SECTION("int: SMALL data set")          { CNAME <int, PARAM> test  ; CHECK(test.run(SMALL_DATA_SET) == 0 );}\
    SECTION("int: MEDIUM data set")         { CNAME <int, PARAM> test  ; CHECK(test.run(MEDIUM_DATA_SET) == 0 );}\
    SECTION("long: SMALL ALIGNED data set") { CNAME <long, PARAM> test ; CHECK(test.run(SMALL_ALIGNED_DATA_SET) == 0 );}\
    SECTION("long: SMALL data set")         { CNAME <long, PARAM> test ; CHECK(test.run(SMALL_DATA_SET) == 0 );}\
    SECTION("long: MEDIUM data set")        { CNAME <long, PARAM> test ; CHECK(test.run(MEDIUM_DATA_SET) == 0 );}\
}

#define RUN_PERF_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " performance test", "[" NAME "][PERF]" ) {\
    SECTION("int: PERF data set")   {\
        CNAME <int, PARAM> test; \
        CHECK(test.run(PERF_DATA_SET, true) == 0 );\
    }\
    SECTION("int: PERF data set")   {\
        CNAME <long, PARAM> test;\
        CHECK(test.run(PERF_DATA_SET, true) == 0 );\
    }\
}

#define RUN_BENCHMARK_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " benchmark test", "[.][" NAME "][BENCHMARK]" ) {\
    long i;\
    SECTION("int: BENCHMARK data set")   {\
        for (i = 1000; i < 1000000; i*=10)\
            CNAME <int, PARAM> ().run(i, true);\
        for (i = 1000000; i< 100000000; i+= 10 * 1000000)\
            CNAME <int, PARAM> ().run(i, true);\
        for (i = 100000000; i<= 250000000; i+= 5 * 10000000)\
            CNAME <int, PARAM> ().run(i, true);\
    }\
    SECTION("long: BENCHMARK data set")   {\
        for (i = 1000; i < 1000000; i*=10)\
            CNAME <long, PARAM> ().run(i, true);\
        for (i = 1000000; i< 100000000; i+= 10 * 1000000)\
            CNAME <long, PARAM> ().run(i, true);\
        for (i = 100000000; i<= 125000000 ; i+= 25 * 1000000)\
            CNAME <long, PARAM> ().run(i, true);\
    }\
}

/* #define RUN_BENCHMARK_TEST(NAME, CNAME, PARAM)\ */
/* TEST_CASE( NAME " benchmark test", "[.][" NAME "][BENCHMARK][BENCHMARK_ALL]" ) {\ */
/*     unsigned long i;\ */
/*     SECTION("long: BENCHMARK data set")   {\ */
/*         CNAME <long, PARAM> test;\ */
/*         CNAME <int, PARAM> testi;\ */
/*         for (i = 250000000 ; i<= 300000256; i+= 10* 5 * 10000000)\ */
/*             CHECK(testi.run(i, true) == 0 );\ */
/*     }\ */
/* } */

#endif /* end of include guard: TEST_BASE_CUH_WT3FRCI9 */

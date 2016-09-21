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

    virtual int run(unsigned long max_size, bool print = false, unsigned int fixed_bit_lenght=0)
    {
        TIMEIT_SETUP();
        pre_setup(max_size);

        int error_count = 0;
        
        unsigned int min_bit_lenght = (unsigned int)getenv_extract_int("GPU_MIN_BIT_LENGTH", 1, cword, 1);
        unsigned int max_bit_lenght = (unsigned int)getenv_extract_int("GPU_MAX_BIT_LENGTH", 1, cword, cword);

        if(fixed_bit_lenght > 0) { // overwrite previous settings, dirty fix for real data
            min_bit_lenght = fixed_bit_lenght;
            max_bit_lenght = fixed_bit_lenght + 1;
        }

        for (unsigned int _bit_lenght = min_bit_lenght; _bit_lenght < max_bit_lenght; ++_bit_lenght) {
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
TEST_CASE( NAME " test set", "[" NAME "][SMALL]") {\
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

#define KB(x) (x) * 1000 
#define MB(x) (x) * 1000 * KB(1)
#define GB(x) (x) * 1000 * MB(1)

#define RUN_BENCHMARK_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " benchmark test", "[.][" NAME "][BENCHMARK]" ) {\
    unsigned long sizes[] = {\
        KB(1), KB(5), KB(10), KB(50), KB(100), KB(250), KB(500), KB(750),\
        MB(1), MB(5), MB(10), MB(50), MB(100), MB(250), MB(500), GB(1)\
    };\
    SECTION("int: BENCHMARK data set")   {\
        for (int i = 0; i < 16; i++){\
            CNAME <int, PARAM> test;\
            CHECK(test.run(sizes[i] / sizeof(int), true) == 0 );\
        }\
    }\
    SECTION("long: BENCHMARK data set")   {\
        for (int i = 0; i < 16; i++){\
            CNAME <long, PARAM> test;\
            CHECK(test.run(sizes[i] / sizeof(long), true) == 0 );\
        }\
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

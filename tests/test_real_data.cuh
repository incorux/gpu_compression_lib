#ifndef TEST_REAL_DATA_H
#define TEST_REAL_DATA_H
#include "catch.hpp"
#include "../compression/macros.cuh"

#include "test_aafl.cuh"
#include "test_afl.cuh"
#include "test_base.cuh"
#include "test_delta.cuh"
#include "test_macros.cuh"
#include "test_pafl.cuh"
#include "test_delta_aafl.cuh"
#include "test_real_data.cuh"

#define RUN_FILE_BENCHMARK_TEST(NAME, CNAME, TPARAM, IPARAM, FILENAME, COMP_PARAMS, SORT)\
TEST_CASE( NAME " real data benchmark test", "[.][" NAME "][REAL]" ) {\
    SECTION("int: BENCHMARK data set")   {\
            CNAME <TPARAM, IPARAM> test;\
            CHECK(test.run_on_file(FILENAME, COMP_PARAMS, true, SORT)==0);\
    }\
}

#endif /* TEST_REAL_DATA_H */

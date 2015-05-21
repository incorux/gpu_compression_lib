#include "test_afl.cuh"
#include "config.cuh"

#define RUN_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " test set", "[" NAME "]" ) {\
    SECTION("int: SMALL ALIGNED data set")   {CNAME <int, PARAM> ().run(SMALL_ALIGNED_DATA_SET);}\
    SECTION("int: SMALL data set")   {CNAME <int, PARAM> ().run(SMALL_DATA_SET);}\
    SECTION("int: MEDIUM data set")  {CNAME <int, PARAM>  ().run(MEDIUM_DATA_SET);}\
    SECTION("long: SMALL ALIGNED data set")  {CNAME <long, PARAM> ().run(SMALL_ALIGNED_DATA_SET);}\
    SECTION("long: SMALL data set")  {CNAME <long, PARAM> ().run(SMALL_DATA_SET);}\
    SECTION("long: MEDIUM data set")  {CNAME <long, PARAM> ().run(MEDIUM_DATA_SET);}\
}

RUN_TEST("AFL", test_afl, 32);
RUN_TEST("FL", test_afl, 1);

RUN_TEST("AFL_CPU", test_afl_cpu, 32);
RUN_TEST("FL_CPU", test_afl_cpu, 1);

RUN_TEST("RAFL_CPU", test_afl_random_access_cpu, 32);
RUN_TEST("RFL_CPU", test_afl_random_access_cpu, 1);

RUN_TEST("RAFL", test_afl_random_access, 32);
RUN_TEST("RFL", test_afl_random_access, 1);

#define RUN_PERF_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " performance test", "[" NAME "][PERF][hide]" ) {\
    SECTION("int: PERF data set")   {CNAME <int, PARAM> ().run(PERF_DATA_SET, true);}\
    SECTION("long: PERF data set")  {CNAME <long, PARAM>  ().run(PERF_DATA_SET, true);}\
}

RUN_PERF_TEST("AFL", test_afl, 32);
RUN_PERF_TEST("FL", test_afl, 1);

RUN_PERF_TEST("RAFL", test_afl_random_access, 32);
RUN_PERF_TEST("RFL", test_afl_random_access, 1);

RUN_PERF_TEST("AFL_CPU", test_afl_cpu, 32);
RUN_PERF_TEST("FL_CPU", test_afl_cpu, 1);

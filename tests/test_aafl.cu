#include "test_aafl.cuh"
#include "config.cuh"

#define RUN_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " test set", "[" NAME "]" ) {\
    SECTION("int: SMALL ALIGNED data set")   {CNAME <int, PARAM> ().run(SMALL_ALIGNED_DATA_SET);}\
    SECTION("int: MEDIUM data set")  {CNAME <int, PARAM>  ().run(MEDIUM_DATA_SET);}\
    SECTION("int: SMALL data set")   {CNAME <int, PARAM> ().run(SMALL_DATA_SET);}\
    SECTION("long: SMALL ALIGNED data set")  {CNAME <long, PARAM> ().run(SMALL_ALIGNED_DATA_SET);}\
    SECTION("long: SMALL data set")  {CNAME <long, PARAM> ().run(SMALL_DATA_SET);}\
    SECTION("long: MEDIUM data set")  {CNAME <long, PARAM> ().run(MEDIUM_DATA_SET);}\
}

RUN_TEST("AAFL", test_aafl, 32);

#define RUN_PERF_TEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " performance test", "[" NAME "][PERF][hide]" ) {\
    SECTION("int: PERF data set")   {CNAME <int, PARAM> ().run(PERF_DATA_SET, true);}\
    SECTION("long: PERF data set")  {CNAME <long, PARAM>  ().run(PERF_DATA_SET, true);}\
}

RUN_PERF_TEST("AAFL", test_aafl, 32);

#include "test_pafl.cuh"
#include "config.cuh"

#define RUN_PTEST(NAME, CNAME, PARAM)\
TEST_CASE( NAME " test set", "[" NAME "]" ) {\
    SECTION("int: SMALL data set")   {CNAME <int, PARAM> (0.1).run(SMALL_DATA_SET);}\
    SECTION("int: MEDIUM data set")  {CNAME <int, PARAM>  (0.1).run(MEDIUM_DATA_SET);}\
}

RUN_PTEST("PAFL", test_pafl, 32);
RUN_PTEST("PFL", test_pafl, 1);

#include "test_delta.cuh"
#include "config.cuh"

#define RUN_TEST_DELTA(NAME, CNAME, PARAM)\
TEST_CASE( NAME " test set", "[" NAME "]" ) {\
    SECTION("int: SMALL ALIGNED data set")   {CNAME <int, PARAM> ().run(SMALL_ALIGNED_DATA_SET);}\
    SECTION("int: MEDIUM data set")  {CNAME <int, PARAM>  ().run(MEDIUM_DATA_SET);}\
}

    /* SECTION("int: MEDIUM data set")  {CNAME <int, PARAM>  ().run(MEDIUM_DATA_SET);}\ */
    /* SECTION("int: SMALL data set")   {CNAME <int, PARAM> ().run(SMALL_DATA_SET);}\ */
    /* SECTION("long: SMALL ALIGNED data set")  {CNAME <long, PARAM> ().run(SMALL_ALIGNED_DATA_SET);}\ */
    /* SECTION("long: SMALL data set")  {CNAME <long, PARAM> ().run(SMALL_DATA_SET);}\ */
    /* SECTION("long: MEDIUM data set")  {CNAME <long, PARAM> ().run(MEDIUM_DATA_SET);}\ */
/* } */

RUN_TEST_DELTA("DELTA_AFL", test_delta, 32)


/* TEST_CASE( " DELTA test set", "[DELTA]" ) { */
/*     SECTION("int: SMALL ALIGNED data set")   {test_delta <int, 32> ().run(SMALL_ALIGNED_DATA_SET);} */
/* } */

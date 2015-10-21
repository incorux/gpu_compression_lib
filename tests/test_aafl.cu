#include "test_aafl.cuh"
#include "config.cuh"

RUN_TEST("AAFL", test_aafl, 32);
RUN_PERF_TEST("AAFL", test_aafl, 32);
RUN_BENCHMARK_TEST("AAFL", test_aafl, 32);

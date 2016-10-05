#include "test_real_data.cuh"

/* RUN_FILE_BENCHMARK_TEST("AFL_CPU", test_afl_cpu, int, 32, "real_data_benchmarks/LINENUMBER.bin", 3, false) */
/* RUN_FILE_BENCHMARK_TEST("AFL", test_afl, int, 32, "real_data_benchmarks/LINENUMBER.bin", 3, false) */
/* RUN_FILE_BENCHMARK_TEST("FL", test_afl, int, 1, "real_data_benchmarks/LINENUMBER.bin", 3, false) */
RUN_FILE_BENCHMARK_TEST("AAFL", test_aafl, int, 32, "real_data_benchmarks/COMMITDATE.bin", 30, true)
RUN_FILE_BENCHMARK_TEST("DELTA_AFL", test_delta, int, 32, "real_data_benchmarks/COMMITDATE.bin", 30, true)
RUN_FILE_BENCHMARK_TEST("DELTA_AAFL", test_delta_aafl, int, 32, "real_data_benchmarks/COMMITDATE.bin", 30, true)
RUN_FILE_BENCHMARK_TEST("DELTA_PAFL", test_delta_pafl, int, 32, "real_data_benchmarks/COMMITDATE.bin", 1, true)

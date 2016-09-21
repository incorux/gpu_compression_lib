#include "test_real_data.cuh"

TEST_CASE("REAL AFL benchmark test", "[.][AFL_CPU][REAL]" ) {
    SECTION("REAL DATA BENCHMARK data set") {
        test_afl_cpu_real_data <int, 32> test;
        CHECK(test.run_on_file("real_data_benchmarks/LINENUMBER.bin", 3) == 0);
    }
}

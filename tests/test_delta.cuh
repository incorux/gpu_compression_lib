#ifndef TEST_DELTA_CUH_ISZ6QCRW
#define TEST_DELTA_CUH_ISZ6QCRW
#include "test_afl.cuh"

template <typename T, int CWARP_SIZE> class test_delta: public test_afl<T, CWARP_SIZE> {
public: virtual void decompressData(int bit_length) {
        run_afl_decompress_value_gpu <T, CWARP_SIZE> (bit_length, this->dev_out, this->dev_data, this->max_size);
    }
};

#endif /* end of include guard: TEST_DELTA_CUH_ISZ6QCRW */

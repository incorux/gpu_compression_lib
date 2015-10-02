#ifndef AAFL_GPU_CUH_NYKUAXIF
#define AAFL_GPU_CUH_NYKUAXIF
#include "compression/macros.cuh"
#include "compression/afl_gpu.cuh"

template <typename T, char CWARP_SIZE>
__device__  void aafl_compress_base_gpu (unsigned long *compressed_data_register, unsigned int *warp_bit_lenght, unsigned long *warp_position_id, unsigned long data_id, T *data, T *compressed_data, unsigned long length);

__inline__ __device__
int warpAllReduceMax(int val) {
    int m = val;
    for (int mask = warpSize/2; mask > 0; mask /= 2) {
        m = __shfl_xor(val, mask);
        val = m > val ? m : val;
    }
    return val;
}

#endif /* end of include guard: AAFL_GPU_CUH_NYKUAXIF */

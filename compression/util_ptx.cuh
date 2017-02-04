#pragma once
#include "util_macro.cuh"
#include "util_bit.cuh"
#include "util_shfl.cuh"

//TODO: move to some other file
template < typename T, char CWARP_SIZE >
__forceinline__ __device__ unsigned long get_data_id()
{
    const unsigned int warp_lane = get_lane_id(CWARP_SIZE);
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    return data_block * CWORD_SIZE(T) + warp_lane;
}

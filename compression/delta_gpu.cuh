#ifndef DELTA_CUH_KKZZHX97
#define DELTA_CUH_KKZZHX97
#include "macros.cuh"


/* template <typename T> __global__ void delta_compress_gpu (T *data, T *compressed_data, T *spoints, unsigned int bit_length, unsigned long length, unsigned long spoints_length); */

/* template <typename T> __global__ void delta_decompress_gpu (T *compressed_data, T *spoints, T *data, unsigned long length, unsigned int spoints_length, int width=32); */

#define WARP_SZ 32
__device__ inline int get_lane_id(void) { return threadIdx.x % WARP_SZ; } //TODO: move to macros and reuse


template < typename T, char CWARP_SIZE > __host__ void run_delta_afl_decompress_gpu(const unsigned int bit_length, T *compressed_data, T *compressed_data_block_start, T *data, unsigned long length);

template < typename T, char CWARP_SIZE > __host__ void run_delta_afl_compress_gpu(const unsigned int bit_length, T *data, T *compressed_data, T* compressed_data_block_start, unsigned long length);

template < typename T, char CWARP_SIZE > __global__ void delta_afl_decompress_gpu (const unsigned int bit_length, T *compressed_data, T* compressed_data_block_start, T * decompress_data, unsigned long length);

template < typename T, char CWARP_SIZE > __global__ void delta_afl_compress_gpu (const unsigned int bit_length, T *data, T *compressed_data, T* compressed_data_block_start, unsigned long length);

template <typename T, char CWARP_SIZE> __device__ void delta_afl_decompress_base_gpu ( const unsigned int bit_length, unsigned long comp_data_id, unsigned long data_id, T *compressed_data, T* compressed_data_block_start, T *data, unsigned long length);

template <typename T, char CWARP_SIZE> __device__  void delta_afl_compress_base_gpu (const unsigned int bit_length, unsigned long data_id, unsigned long comp_data_id, T *data, T *compressed_data, T* compressed_data_block_start, unsigned long length);
#endif /* end of include guard: DELTA_CUH_KKZZHX97 */

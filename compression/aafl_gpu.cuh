#ifndef AAFL_GPU_CUH_NYKUAXIF
#define AAFL_GPU_CUH_NYKUAXIF
#include "compression/macros.cuh"
#include "compression/afl_gpu.cuh"

template <typename T, char CWARP_SIZE>
__device__  void aafl_compress_base_gpu (unsigned long *compressed_data_register, unsigned char *warp_bit_lenght, unsigned long *warp_position_id, unsigned long data_id, T *data, T *compressed_data, unsigned long length);


template < typename T , char CWARP_SIZE > __host__ void run_aafl_decompress_gpu ( unsigned char *warp_bit_lenght, unsigned long *warp_position_id, T *data            , T *compressed_data   , unsigned long length);
template < typename T , char CWARP_SIZE > __host__ void run_aafl_compress_gpu   (unsigned long *compressed_data_register, unsigned char *warp_bit_lenght, unsigned long *warp_position_id, T *compressed_data , T *decompressed_data , unsigned long length);

template < typename T , char CWARP_SIZE > __global__ void aafl_compress_gpu     ( unsigned long *compressed_data_register, unsigned char *warp_bit_lenght, unsigned long *warp_position_id, T *data            , T *compressed_data   , unsigned long length);
template < typename T , char CWARP_SIZE > __global__ void aafl_decompress_gpu   ( unsigned char *warp_bit_lenght, unsigned long *warp_position_id, T *compressed_data , T * decompress_data  , unsigned long length);

#endif /* end of include guard: AAFL_GPU_CUH_NYKUAXIF */

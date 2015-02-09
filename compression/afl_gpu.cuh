#ifndef afl_H
#define afl_H 1

__global__ void afl_compress_gpu        ( int bit_length,  int *data,             int *compressed_data,   unsigned long length);
__global__ void afl_decompress_gpu      ( int bit_length,  int *compressed_data,  int * decompress_data,  unsigned long length);

__device__ void afl_compress_base_gpu   ( int bit_length,  unsigned long data_id,       unsigned long comp_data_id,  int *data,              int * compressed_data,  unsigned long length);
__device__ void afl_decompress_base_gpu ( int bit_length,  unsigned long comp_data_id,  unsigned long data_id,       int *compressed_data,   int *data,              unsigned long length);

__host__ void run_afl_decompress_gpu    ( int bit_length,  int *data,                   int *compressed_data,        unsigned long length);
__host__ void run_afl_compress_gpu      ( int bit_length,  int *compressed_data,        int *decompressed_data,      unsigned long length);

__device__ int afl_decompress_base_value_gpu (int bit_length, int *compressed_data, unsigned long pos);

__global__ void afl_decompress_value_gpu (int bit_length, int *compressed_data, int * decompress_data, unsigned long length);

__host__ void run_afl_decompress_value_gpu(int bit_length, int *compressed_data, int *data, unsigned long length);
#endif

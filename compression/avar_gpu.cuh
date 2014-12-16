#ifndef AVAR_H
#define AVAR_H 1
typedef struct avar_header
{
    unsigned int bit_length;
    unsigned int warp_size;
    unsigned int word_size;
} avar_header;

__global__ void avar_compress_gpu (avar_header comp_h, int *data, int *compressed_data, long length);

__global__ void avar_decompress_gpu (avar_header comp_h, int *compressed_data, int * decompress_data, long length);

__device__ void avar_compress_base_gpu (avar_header comp_h, long data_id, long comp_data_id, int *data, int * compressed_data, long length);

__device__ void avar_decompress_base_gpu (avar_header comp_h, long comp_data_id, long data_id, int *compressed_data, int *data, long length);

__host__ void run_avar_decompress_gpu(avar_header comp_h, int *data, int *compressed_data, long length);
__host__ void run_avar_compress_gpu(avar_header comp_h, int *compressed_data, int *decompressed_data, long length);
#endif

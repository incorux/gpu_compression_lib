#ifndef AVAR_H
#define AVAR_H 1
typedef struct avar_header
{
    unsigned int bit_length;
} avar_header;

__global__ void avar_compress_gpu (avar_header comp_h, int *data, int *compressed_data, size_t length);

__global__ void avar_decompress_gpu (avar_header comp_h, int *compressed_data, int * decompress_data, size_t length);

__device__ void avar_compress_base_gpu (avar_header comp_h, size_t data_id, size_t comp_data_id, int *data, int * compressed_data, size_t length);

__device__ void avar_decompress_base_gpu (avar_header comp_h, size_t comp_data_id, size_t data_id, int *compressed_data, int *data, size_t length);

__host__ void run_avar_decompress_gpu(avar_header comp_h, int *data, int *compressed_data, size_t length);
__host__ void run_avar_compress_gpu(avar_header comp_h, int *compressed_data, int *decompressed_data, size_t length);
#endif

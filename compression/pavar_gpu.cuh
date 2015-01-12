#ifndef AVAR_H
#define AVAR_H 1
typedef struct pavar_header
{
    unsigned int bit_length;
    unsigned int patch_bit_length;
} pavar_header;

__global__ void pavar_compress_gpu (pavar_header comp_h, int *data, int *compressed_data, unsigned long length);

__global__ void pavar_decompress_gpu (pavar_header comp_h, int *compressed_data, int * decompress_data, unsigned long length);

__device__ void pavar_compress_base_gpu (pavar_header comp_h, unsigned long data_id, unsigned long comp_data_id, int *data, int * compressed_data, unsigned long length);

__device__ void pavar_decompress_base_gpu (pavar_header comp_h, unsigned long comp_data_id, unsigned long data_id, int *compressed_data, int *data, unsigned long length);

__host__ void run_pavar_decompress_gpu(pavar_header comp_h, int *data, int *compressed_data, unsigned long length);
__host__ void run_pavar_compress_gpu(pavar_header comp_h, int *compressed_data, int *decompressed_data, unsigned long length);
#endif

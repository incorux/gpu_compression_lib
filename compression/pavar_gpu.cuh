#ifndef PAVAR_H
#define PAVAR_H 1
typedef struct pavar_header
{
    unsigned int bit_length;
    unsigned int patch_bit_length;
} pavar_header;

__global__ void pavar_compress_gpu (
        pavar_header comp_h, 
        int *data, 
        int *compressed_data, 
        unsigned long length,
        
        int *global_queue_patch_values,
        int *global_queue_patch_index,
        int *global_queue_patch_count,

        int *global_data_patch_values,
        int *global_data_patch_index,
        int *global_data_patch_count
        );

__global__ void pavar_decompress_gpu (pavar_header comp_h, int *compressed_data, int * decompress_data, unsigned long length);

__device__ void pavar_compress_base_gpu (
        pavar_header comp_h, 

        unsigned long data_id, 
        unsigned long comp_data_id, 

        int *data, 
        int *compressed_data, 
        unsigned long length,

        int *private_patch_values,
        int *private_patch_index,
        int *private_patch_count,

        int *global_patch_values,
        int *global_patch_index,
        int *global_patch_count
        );

__device__ void pavar_decompress_base_gpu (pavar_header comp_h, unsigned long comp_data_id, unsigned long data_id, int *compressed_data, int *data, unsigned long length);

__host__ void run_pavar_decompress_gpu(pavar_header comp_h, int *data, int *compressed_data, unsigned long length);

__host__ void run_pavar_compress_gpu(
        pavar_header comp_h, 
        int *data, 
        int *compressed_data, 
        unsigned long length,
        
        int *global_queue_patch_values,
        int *global_queue_patch_index,
        int *global_queue_patch_count,

        int *global_data_patch_values,
        int *global_data_patch_index,
        int *global_data_patch_count
        );
#endif

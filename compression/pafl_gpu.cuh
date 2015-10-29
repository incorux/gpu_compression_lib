#ifndef Pafl_H
#define Pafl_H 1
typedef struct pafl_header
{
    unsigned int bit_length;
    unsigned int patch_bit_length;
} pafl_header;

template <typename T, char CWARP_SIZE> 
__global__ void pafl_compress_gpu (
        pafl_header comp_h, 
        T *data, 
        T *compressed_data, 
        unsigned long length,
        
        T *global_queue_patch_values,
        T *global_queue_patch_index,
        T *global_queue_patch_count,

        T *global_data_patch_values,
        T *global_data_patch_index,
        T *global_data_patch_count
        );

template <typename T, char CWARP_SIZE> 
__global__ void pafl_decompress_gpu (pafl_header comp_h, int *compressed_data, int * decompress_data, unsigned long length);

template <typename T, char CWARP_SIZE> 
__device__ void pafl_compress_base_gpu (
        pafl_header comp_h, 

        unsigned long data_id, 
        unsigned long comp_data_id, 

        T *data, 
        T *compressed_data, 
        unsigned long length,

        T *private_patch_values,
        T *private_patch_index,
        T *private_patch_count,

        T *global_patch_values,
        T *global_patch_index,
        T *global_patch_count
        );

template <typename T, char CWARP_SIZE> 
__device__ void pafl_decompress_base_gpu (pafl_header comp_h, unsigned long comp_data_id, unsigned long data_id, T *compressed_data, T *data, unsigned long length);

template <typename T, char CWARP_SIZE> 
__host__ void run_pafl_decompress_gpu(
        pafl_header comp_h, 
        T *compressed_data, 
        T *data, 
        unsigned long length,
        
        T *global_data_patch_values,
        T *global_data_patch_index,
        unsigned long *global_data_patch_count
        );

template <typename T, char CWARP_SIZE> 
__host__ void run_pafl_compress_gpu(
        pafl_header comp_h, 
        T *data, 
        T *compressed_data, 
        unsigned long length,
        
        T *global_queue_patch_values,
        T *global_queue_patch_index,
        T *global_queue_patch_count,

        T *global_data_patch_values,
        T *global_data_patch_index,
        T *global_data_patch_count
        );

template <typename T, char CWARP_SIZE> 
__host__ void run_pafl_compress_gpu_alternate(
        pafl_header comp_h, 
        T *data, 
        T *compressed_data, 
        unsigned long length,
        
        T *global_queue_patch_values,
        T *global_queue_patch_index,
        unsigned long *global_queue_patch_count,

        T *global_data_patch_values,
        T *global_data_patch_index,
        unsigned long *global_data_patch_count
        );

#endif

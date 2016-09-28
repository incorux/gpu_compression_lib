#ifndef Pafl_H
#define Pafl_H 1

template <typename T, char CWARP_SIZE> 
__host__ void run_pafl_decompress_gpu(
        unsigned int bit_length,
        T *compressed_data, 
        T *data, 
        unsigned long length,
        
        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        );


template <typename T, char CWARP_SIZE> 
__host__ void run_pafl_compress_gpu_alternate(
        unsigned int bit_length,
        T *data, 
        T *compressed_data, 
        unsigned long length,

        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        );

template <typename T, char CWARP_SIZE> 
__host__ void run_delta_pafl_decompress_gpu(
        unsigned int bit_length,
        T *compressed_data, 
        T* compressed_data_block_start,
        T *data, 
        unsigned long length,
        
        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        );

template <typename T, char CWARP_SIZE> 
__host__ void run_delta_pafl_compress_gpu_alternate(
        unsigned int bit_length,
        T *data, 
        T *compressed_data, 
        T* compressed_data_block_start,
        unsigned long length,

        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        );

#endif

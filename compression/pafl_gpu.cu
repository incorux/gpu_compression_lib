#include "pafl_gpu.cuh"
#include "afl_gpu.cuh"
#include "macros.cuh"
#include "../tools/cuda.cuh"

#include <math.h>
#include <stdio.h>

#define BIT_SET(a,b) ((a) |= (1UL<<(b)))
#define BIT_CLEAR(a,b) ((a) &= ~(1UL<<(b)))
#define BIT_FLIP(a,b) ((a) ^= (1UL<<(b)))
#define BIT_CHECK(a,b) ((a) & (1UL<<(b)))


template <typename T, char CWARP_SIZE>
__device__  void pafl_compress_base_gpu3 (
        const unsigned int bit_length, 
        unsigned long data_id, unsigned long comp_data_id,
        T *data, T *compressed_data,
        unsigned long length,

        T *global_patch_values,
        unsigned long *global_patch_index,
        unsigned long *global_patch_count
        )
{
    T v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;
    int exception_counter = 0;

    T exception_buffer[8];
    unsigned long position_mask = 0;

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_data < length; ++i) 
    {
        v1 = data[pos_data];

        if(BITLEN(v1) >= bit_length){
            exception_buffer[exception_counter] = v1;
            exception_counter ++;
            BIT_SET(position_mask, i);
        }

        pos_data += CWARP_SIZE;

        if (v1_pos >= CWORD_SIZE(T) - bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);

            compressed_data[pos] = value;

            v1_pos = bit_length - v1_len;
            value = GETNPBITS(v1, v1_pos, v1_len); 

            pos += CWARP_SIZE;  
        } else {
            v1_len = bit_length;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);
            v1_pos += v1_len;
        }
    }
    if (pos_data >= length  && pos_data < length + CWARP_SIZE)
    {
        compressed_data[pos] = value;
    }

    int lane_id = get_lane_id();
    unsigned long local_counter = 0;

    int warp_exception_counter = shfl_prefix_sum(exception_counter);

    if(lane_id == 31 && warp_exception_counter > 0){
        local_counter = atomicAdd((unsigned long long int *)global_patch_count, (unsigned long long int)warp_exception_counter);
    }

    local_counter = shfl_get_value((long)local_counter, 31);
    
    for (int i = 0; i < exception_counter; ++i)
        global_patch_values[local_counter + warp_exception_counter - exception_counter + i] = exception_buffer [i]; 

    for (unsigned int i = 0, j = 0; i < exception_counter && j < CWORD_SIZE(T); j++){
        if (BIT_CHECK(position_mask, j)) {
            global_patch_index[local_counter + warp_exception_counter - exception_counter + i] = data_id + j * CWARP_SIZE; 
            i++;
        }
    }
}

template < typename T, char CWARP_SIZE >
__global__ void pafl_compress_gpu3 (
        const unsigned int bit_length, 
        T *data, T *compressed_data, 
        unsigned long length,
        
        T *global_queue_patch_values,
        unsigned long *global_queue_patch_index,
        unsigned long *global_queue_patch_count
        )
{
    const unsigned int warp_lane = threadIdx.x % CWARP_SIZE; 
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    pafl_compress_base_gpu3 <T, CWARP_SIZE> (
            bit_length, data_id, cdata_id, data, compressed_data,
            length,
            global_queue_patch_values,
            global_queue_patch_index, 
            global_queue_patch_count);
}

template <typename T, char CWARP_SIZE> 
__global__ void patch_apply_gpu3 (
        pafl_header comp_h,
        T *decompressed_data,
        unsigned long length,
        
        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        ) //TODO: fix params list
{
    unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long patch_length = *global_data_patch_count;

    if (tid < patch_length)
    {
        unsigned long idx = global_data_patch_index[tid];
        T val = global_data_patch_values[tid];
        decompressed_data[idx] = val;
    }
}

template <typename T, char CWARP_SIZE> 
__host__ void run_pafl_compress_gpu_alternate(
        pafl_header comp_h, 
        T *data, 
        T *compressed_data, 
        unsigned long length,
        
        T *global_queue_patch_values,
        unsigned long *global_queue_patch_index,
        unsigned long *global_queue_patch_count,

        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        )
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy 
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    pafl_compress_gpu3 <T, CWARP_SIZE> <<<block_number, block_size>>> (
            comp_h.bit_length, 
            data, 
            compressed_data, 
            length,

            global_data_patch_values,
            global_data_patch_index,
            global_data_patch_count
            );


    cudaErrorCheck();
}


template <typename T, char CWARP_SIZE> 
__host__ void run_pafl_decompress_gpu(
        pafl_header comp_h, 
        T *compressed_data, 
        T *data, 
        unsigned long length,
        
        T *global_data_patch_values,
        unsigned long *global_data_patch_index,
        unsigned long *global_data_patch_count
        )
{
    int block_size = CWARP_SIZE * 8; // better occupancy
    unsigned long block_number = (length + block_size * CWARP_SIZE - 1) / (block_size * CWARP_SIZE);
    afl_decompress_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (comp_h.bit_length, compressed_data, data, length);

    cudaErrorCheck();
    
    patch_apply_gpu3 <T, CWARP_SIZE> <<<block_number * CWARP_SIZE, block_size>>> (
            comp_h, 
            data, 
            length,

            global_data_patch_values,
            global_data_patch_index,
            global_data_patch_count
            );
}

#define GFL_SPEC(X, A) \
template __host__ void run_pafl_compress_gpu_alternate <X,A> ( pafl_header , X *, X *, unsigned long , X *, unsigned long *, unsigned long *, X *, unsigned long *, unsigned long *);\
template __host__ void run_pafl_decompress_gpu <X,A> ( pafl_header , X *, X *, unsigned long , X *, unsigned long *, unsigned long *);

#define AFL_SPEC(X) GFL_SPEC(X, 32)
FOR_EACH(AFL_SPEC, int, long)

#define FL_SPEC(X) GFL_SPEC(X, 1)
FOR_EACH(FL_SPEC, int, long)

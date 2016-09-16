#include "aafl_gpu.cuh"
#include <stdio.h>

template < typename T, char CWARP_SIZE >
__host__ void run_aafl_compress_gpu(unsigned long *compressed_data_register, unsigned char *warp_bit_lenght, unsigned long *warp_position_id, T *data, T *compressed_data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy 
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    aafl_compress_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (compressed_data_register, warp_bit_lenght, warp_position_id, data, compressed_data, length);
}

template < typename T, char CWARP_SIZE >
__host__ void run_aafl_decompress_gpu(unsigned char *warp_bit_lenght, unsigned long *warp_position_id, T *compressed_data, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    aafl_decompress_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (warp_bit_lenght, warp_position_id, compressed_data, data, length);
}

template < typename T, char CWARP_SIZE >
__global__ void aafl_compress_gpu (unsigned long *compressed_data_register, unsigned char *warp_bit_lenght, unsigned long *warp_position_id, T *data, T *compressed_data, unsigned long length)
{
    const unsigned int warp_lane = (threadIdx.x % CWARP_SIZE); 
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;

    aafl_compress_base_gpu <T, CWARP_SIZE> (compressed_data_register, warp_bit_lenght, warp_position_id, data_id, data, compressed_data, length);
}

template < typename T, char CWARP_SIZE >
__global__ void aafl_decompress_gpu (unsigned char *warp_bit_lenght, unsigned long *warp_position_id, T *compressed_data, T * decompress_data, unsigned long length)
{
    const unsigned int warp_lane = (threadIdx.x % CWARP_SIZE); 
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;

    if (data_id >= length) return;

    const unsigned long data_block_mem = (blockIdx.x * blockDim.x) / CWARP_SIZE  + threadIdx.x / CWARP_SIZE;
    //TODO: check if shfl propagation is faster
    /* if(data_block_mem * 32 * sizeof(T) * 8 < length) { */
        unsigned long comp_data_id = warp_position_id[data_block_mem] + warp_lane;
        unsigned int bit_length = warp_bit_lenght[data_block_mem];

        if(bit_length > 0)
            afl_decompress_base_gpu <T, CWARP_SIZE> (bit_length, comp_data_id, data_id, compressed_data, decompress_data, length);
        else
            afl_decompress_zeros_gpu <T, CWARP_SIZE> (bit_length, comp_data_id, data_id, compressed_data, decompress_data, length);
    /* } */
}


template <typename T, char CWARP_SIZE>
__device__  void aafl_compress_base_gpu (unsigned long *compressed_data_register, unsigned char *warp_bit_lenght, unsigned long *warp_position_id, unsigned long data_id, T *data, T *compressed_data, unsigned long length)
{

    unsigned long pos_data = data_id;

    unsigned int bit_length = 0, i = 0;
    
    T max_val = 0;

    // Compute bit length for compressed block of data
    for (i = 0; i < CWORD_SIZE(T) && pos_data < length; ++i) 
    {
        max_val = data[pos_data] > max_val ?  data[pos_data] : max_val;
        pos_data += CWARP_SIZE;
    }

    i = warpAllReduceMax(i); 
    // Warp vote for maximum bit length
    bit_length = max_val > 0 ? BITLEN(max_val) + 1 : 0;
    bit_length = warpAllReduceMax(bit_length);

    // Skip if i == 0, i.e. if all values in block are set to 0
    if (i > 0) { 

        // leader thread registers memory in global
        unsigned long comp_data_id = 0;

        if (threadIdx.x % CWARP_SIZE == 0) { 
            const unsigned long data_block = (blockIdx.x * blockDim.x) / CWARP_SIZE + threadIdx.x / CWARP_SIZE;
            unsigned long long int space = bit_length * CWARP_SIZE;

            if(data_id + CWARP_SIZE * CWORD_SIZE(T) > length && data_id < length) {
                space = (( (length - data_id + CWORD_SIZE(T) - 1) / CWORD_SIZE(T)) * bit_length + CWARP_SIZE - 1) / CWARP_SIZE;
                space *= CWARP_SIZE;
            }
            
            if (space  > CWARP_SIZE * bit_length) 
                printf("%d %d %ld \n", bit_length, CWARP_SIZE * bit_length, space);

            comp_data_id = (unsigned long long int) atomicAdd( (unsigned long long int *) compressed_data_register, space);
            warp_bit_lenght[data_block] = bit_length;
            warp_position_id[data_block] = comp_data_id;
        }

        if (bit_length > 0) {
            const unsigned int warp_lane = (threadIdx.x % CWARP_SIZE); 
            // Propagate in warp position of compressed block
            comp_data_id = warpAllReduceMax(comp_data_id);
            comp_data_id += warp_lane;

            // Compress using AFL algorithm
            afl_compress_base_gpu <T, CWARP_SIZE> (bit_length, data_id, comp_data_id, data, compressed_data, length);
        }
    }
}

// For now only those versions are available and will be compiled and linked
// This is intentional !!
#define GFL_SPEC(X, A) \
    template __device__  void aafl_compress_base_gpu <X, A> (unsigned long *compressed_data_register, unsigned char *warp_bit_lenght, unsigned long *warp_position_id, unsigned long data_id, X *data, X *compressed_data, unsigned long length);\
    template  __host__ void run_aafl_decompress_gpu <X,A> ( unsigned char *warp_bit_lenght, unsigned long *warp_position_id, X *data            , X *compressed_data   , unsigned long length);\
    template  __host__ void run_aafl_compress_gpu   <X,A> (unsigned long *compressed_data_register, unsigned char *warp_bit_lenght, unsigned long *warp_position_id, X *compressed_data , X *decompressed_data , unsigned long length);\
    template  __global__ void aafl_compress_gpu     <X,A> ( unsigned long *compressed_data_register, unsigned char *warp_bit_lenght, unsigned long *warp_position_id, X *data            , X *compressed_data   , unsigned long length);\
    template  __global__ void aafl_decompress_gpu   <X,A> ( unsigned char *warp_bit_lenght, unsigned long *warp_position_id, X *compressed_data , X * decompress_data  , unsigned long length);

#define AFL_SPEC(X) GFL_SPEC(X, 32)
FOR_EACH(AFL_SPEC, int, long, unsigned int, unsigned long)


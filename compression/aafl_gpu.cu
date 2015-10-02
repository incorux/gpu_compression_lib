#include "aafl_gpu.cuh"

template < typename T, char CWARP_SIZE >
__host__ void run_aafl_compress_gpu(unsigned long *compressed_data_register, unsigned int *warp_bit_lenght, unsigned long *warp_position_id, T *data, T *compressed_data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy 
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    aafl_compress_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (compressed_data_register, warp_bit_lenght, warp_position_id, data, compressed_data, length);
}

template < typename T, char CWARP_SIZE >
__host__ void run_aafl_decompress_gpu(unsigned int *warp_bit_lenght, unsigned long *warp_position_id, T *compressed_data, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    aafl_decompress_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (warp_bit_lenght, warp_position_id, compressed_data, data, length);
}

template < typename T, char CWARP_SIZE >
__global__ void aafl_compress_gpu (unsigned long *compressed_data_register, unsigned int *warp_bit_lenght, unsigned long *warp_position_id, T *data, T *compressed_data, unsigned long length)
{
    const unsigned int warp_lane = (threadIdx.x % CWARP_SIZE); 
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;

    aafl_compress_base_gpu <T, CWARP_SIZE> (compressed_data_register, warp_bit_lenght, warp_position_id, data_id, data, compressed_data, length);
}

template < typename T, char CWARP_SIZE >
__global__ void aafl_decompress_gpu (unsigned int *warp_bit_lenght, unsigned long *warp_position_id, T *compressed_data, T * decompress_data, unsigned long length)
{
    const unsigned int warp_lane = (threadIdx.x % CWARP_SIZE); 
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;

    //TODO: check if shfl propagation is faster
    unsigned long comp_data_id = warp_bit_lenght[data_id];
    unsigned int bit_length = warp_position_id[data_id];

    afl_decompress_base_gpu <T, CWARP_SIZE> (bit_length, comp_data_id, data_id, compressed_data, decompress_data, length);
}


template <typename T, char CWARP_SIZE>
__device__  void aafl_compress_base_gpu (unsigned long *compressed_data_register, unsigned int *warp_bit_lenght, unsigned long *warp_position_id, unsigned long data_id, T *data, T *compressed_data, unsigned long length)
{
    unsigned long pos_data=data_id;
    unsigned int bit_length = 0, tmp_bit_length;

    // Compute bit length for compressed block of data
    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_data < length; ++i) 
    {
        tmp_bit_length = BITLEN(data[pos_data]);

        if (bit_length < tmp_bit_length) 
            bit_length = tmp_bit_length;

        pos_data += CWARP_SIZE;
    }

    // Warp vote for maximum bit length
    bit_length = warpAllReduceMax(bit_length);

    // Select leader
    int mask = __ballot(1);  // mask of active lanes
    int leader = __ffs(mask) - 1;  // -1 for 0-based indexing

    // leader thread registers memory in global
    unsigned long comp_data_id = 0;
    if (leader == threadIdx.x % 32) {
        comp_data_id = (unsigned long long int)atomicAdd( (unsigned long long int *)compressed_data_register, (unsigned long long int)(bit_length * CWARP_SIZE * CWORD_SIZE(T)));
        warp_bit_lenght[data_id] = bit_length;
        warp_position_id[data_id] = comp_data_id;
    }

    // Propagate in warp position of compressed block
    comp_data_id = warpAllReduceMax(comp_data_id);

    // Compress using AFL algorithm
    afl_compress_base_gpu <T, CWARP_SIZE> (bit_length, data_id, comp_data_id, data, compressed_data, length);
}

// For now only those versions are available and will be compiled and linked
// This is intentional !!
#define GFL_SPEC(X, A) \
    template __device__  void aafl_compress_base_gpu <X, A> (unsigned long *compressed_data_register, unsigned int *warp_bit_lenght, unsigned long *warp_position_id, unsigned long data_id, X *data, X *compressed_data, unsigned long length);\
    template  __host__ void run_aafl_decompress_gpu <X,A> ( unsigned int *warp_bit_lenght, unsigned long *warp_position_id, X *data            , X *compressed_data   , unsigned long length);\
    template  __host__ void run_aafl_compress_gpu   <X,A> (unsigned long *compressed_data_register, unsigned int *warp_bit_lenght, unsigned long *warp_position_id, X *compressed_data , X *decompressed_data , unsigned long length);\
    template  __global__ void aafl_compress_gpu     <X,A> ( unsigned long *compressed_data_register, unsigned int *warp_bit_lenght, unsigned long *warp_position_id, X *data            , X *compressed_data   , unsigned long length);\
    template  __global__ void aafl_decompress_gpu   <X,A> ( unsigned int *warp_bit_lenght, unsigned long *warp_position_id, X *compressed_data , X * decompress_data  , unsigned long length);

#define AFL_SPEC(X) GFL_SPEC(X, 32)
FOR_EACH(AFL_SPEC, int, long, unsigned int, unsigned long)


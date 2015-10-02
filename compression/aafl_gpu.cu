#include "aafl_gpu.cuh"

template <typename T, char CWARP_SIZE>
__device__  void aafl_compress_base_gpu (unsigned long *compressed_data_register, unsigned int *warp_bit_lenght, unsigned long *warp_position_id, unsigned long data_id, T *data, T *compressed_data, unsigned long length)
{
    unsigned long pos_data=data_id;
    unsigned int bit_length = 0, tmp_bit_length;
    unsigned int warp_id = 0; //TODO

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
        comp_data_id = atomicAdd(compressed_data_register, bit_length * CWARP_SIZE * CWORD_SIZE(T));
        warp_bit_lenght[warp_id] = bit_length;
        warp_position_id[warp_id] = comp_data_id;
    }

    // Propagate in warp position of compressed block
    comp_data_id = warpAllReduceMax(comp_data_id);

    // Compress using AFL algorithm
    afl_compress_base_gpu <T, CWARP_SIZE> (bit_length, data_id, comp_data_id, data, compressed_data, length);
}
